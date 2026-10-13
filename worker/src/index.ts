import { AwsClient } from "aws4fetch";

export interface Env {
  DB: D1Database;
  BUCKET: R2Bucket;
  API_KEY: string;
  R2_BUCKET_NAME: string;
  R2_ACCOUNT_ID?: string;
  R2_ACCESS_KEY_ID?: string;
  R2_SECRET_ACCESS_KEY?: string;
}

const SLUG_RE = /^[a-z0-9][a-z0-9-]{1,62}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const CONTENT_TYPES: Record<string, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};
const BBOX_FORMATS = new Set(["yolo", "coco", "xyxy"]);
const PRESIGN_TTL_SECONDS = 300;
const MAX_DIRECT_UPLOAD_BYTES = 25 * 1024 * 1024;
const MAX_DETECTIONS = 1000;
const MAX_TAGS_BYTES = 8192;

class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });

async function checkAuth(request: Request, env: Env): Promise<void> {
  if (!env.API_KEY) throw new HttpError(500, "API_KEY secret is not configured");
  const header = request.headers.get("authorization") ?? "";
  const given = header.startsWith("Bearer ") ? header.slice(7) : "";
  // Hash both sides so the comparison is constant-time regardless of length.
  const enc = new TextEncoder();
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(given)),
    crypto.subtle.digest("SHA-256", enc.encode(env.API_KEY)),
  ]);
  if (!crypto.subtle.timingSafeEqual(a, b)) throw new HttpError(401, "unauthorized");
}

// ---------- validation ----------

type Detection = {
  class?: string;
  class_id?: number;
  confidence?: number;
  bbox?: [number, number, number, number];
};

type SampleMeta = {
  id: string;
  content_type: string;
  device_id: string | null;
  model_version: string | null;
  image_width: number | null;
  image_height: number | null;
  bbox_format: string | null;
  captured_at: string | null;
  detections: Detection[] | null;
  tags: Record<string, unknown> | null;
};

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

function optString(body: Record<string, unknown>, key: string, max = 128): string | null {
  const v = body[key];
  if (v === undefined || v === null) return null;
  if (typeof v !== "string" || v.length > max) throw new HttpError(400, `${key} must be a string of at most ${max} chars`);
  return v;
}

function optPositiveInt(body: Record<string, unknown>, key: string): number | null {
  const v = body[key];
  if (v === undefined || v === null) return null;
  if (!Number.isInteger(v) || (v as number) <= 0) throw new HttpError(400, `${key} must be a positive integer`);
  return v as number;
}

function validateDetection(d: unknown, i: number, format: string): Detection {
  if (!isObj(d)) throw new HttpError(400, `detections[${i}] must be an object`);
  const out: Detection = {};
  if (d.class !== undefined && d.class !== null) {
    if (typeof d.class !== "string" || d.class.length === 0 || d.class.length > 128)
      throw new HttpError(400, `detections[${i}].class must be a non-empty string`);
    out.class = d.class;
  }
  if (d.class_id !== undefined && d.class_id !== null) {
    if (!Number.isInteger(d.class_id) || (d.class_id as number) < 0)
      throw new HttpError(400, `detections[${i}].class_id must be a non-negative integer`);
    out.class_id = d.class_id as number;
  }
  if (d.confidence !== undefined && d.confidence !== null) {
    const c = d.confidence;
    if (typeof c !== "number" || !Number.isFinite(c) || c < 0 || c > 1)
      throw new HttpError(400, `detections[${i}].confidence must be a number in [0, 1]`);
    out.confidence = c;
  }
  if (d.bbox !== undefined && d.bbox !== null) {
    const b = d.bbox;
    if (!Array.isArray(b) || b.length !== 4 || !b.every((n) => typeof n === "number" && Number.isFinite(n)))
      throw new HttpError(400, `detections[${i}].bbox must be 4 finite numbers`);
    const [p0, p1, p2, p3] = b as number[];
    if (b.some((n) => n < 0)) throw new HttpError(400, `detections[${i}].bbox must not be negative`);
    if (format === "yolo" && b.some((n) => n > 1))
      throw new HttpError(400, `detections[${i}].bbox must be normalized to [0, 1] for bbox_format=yolo`);
    if (format === "xyxy" && (p2 < p0 || p3 < p1))
      throw new HttpError(400, `detections[${i}].bbox must satisfy x2 >= x1 and y2 >= y1 for bbox_format=xyxy`);
    out.bbox = [p0, p1, p2, p3];
  }
  return out;
}

function validateMeta(body: Record<string, unknown>): SampleMeta {
  let id: string;
  if (body.id === undefined || body.id === null) {
    id = crypto.randomUUID();
  } else if (typeof body.id === "string" && UUID_RE.test(body.id.toLowerCase())) {
    id = body.id.toLowerCase();
  } else {
    throw new HttpError(400, "id must be a UUID");
  }

  const contentType = optString(body, "content_type") ?? "image/jpeg";
  if (!(contentType in CONTENT_TYPES))
    throw new HttpError(400, `content_type must be one of ${Object.keys(CONTENT_TYPES).join(", ")}`);

  let capturedAt = optString(body, "captured_at", 64);
  if (capturedAt !== null) {
    const t = Date.parse(capturedAt);
    if (Number.isNaN(t)) throw new HttpError(400, "captured_at must be an ISO-8601 timestamp");
    capturedAt = new Date(t).toISOString();
  }

  let detections: Detection[] | null = null;
  let bboxFormat = optString(body, "bbox_format");
  if (bboxFormat !== null && !BBOX_FORMATS.has(bboxFormat))
    throw new HttpError(400, "bbox_format must be one of yolo, coco, xyxy");
  if (body.detections !== undefined && body.detections !== null) {
    if (!Array.isArray(body.detections)) throw new HttpError(400, "detections must be an array");
    if (body.detections.length > MAX_DETECTIONS)
      throw new HttpError(400, `detections may contain at most ${MAX_DETECTIONS} items`);
    const fmt = bboxFormat ?? "yolo";
    detections = body.detections.map((d, i) => validateDetection(d, i, fmt));
    if (detections.some((d) => d.bbox)) bboxFormat = fmt;
  }

  let tags: Record<string, unknown> | null = null;
  if (body.tags !== undefined && body.tags !== null) {
    if (!isObj(body.tags)) throw new HttpError(400, "tags must be an object");
    if (JSON.stringify(body.tags).length > MAX_TAGS_BYTES)
      throw new HttpError(400, `tags must serialize to at most ${MAX_TAGS_BYTES} bytes`);
    tags = body.tags;
  }

  return {
    id,
    content_type: contentType,
    device_id: optString(body, "device_id"),
    model_version: optString(body, "model_version"),
    image_width: optPositiveInt(body, "image_width"),
    image_height: optPositiveInt(body, "image_height"),
    bbox_format: bboxFormat,
    captured_at: capturedAt,
    detections,
    tags,
  };
}

// ---------- R2 access ----------

function presignEnabled(env: Env): boolean {
  return Boolean(env.R2_ACCOUNT_ID && env.R2_ACCESS_KEY_ID && env.R2_SECRET_ACCESS_KEY);
}

async function presignPut(env: Env, key: string, contentType: string): Promise<string> {
  const client = new AwsClient({
    accessKeyId: env.R2_ACCESS_KEY_ID!,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY!,
    service: "s3",
    region: "auto",
  });
  const url = new URL(
    `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/${env.R2_BUCKET_NAME}/${key}`,
  );
  url.searchParams.set("X-Amz-Expires", String(PRESIGN_TTL_SECONDS));
  const signed = await client.sign(
    new Request(url, { method: "PUT", headers: { "content-type": contentType } }),
    { aws: { signQuery: true } },
  );
  return signed.url;
}

// ---------- handlers ----------

type Row = {
  id: string;
  project: string;
  image_key: string;
  content_type: string | null;
  uploaded: number;
  archived: number;
  detections: string | null;
  tags: string | null;
  [k: string]: unknown;
};

async function getRow(env: Env, project: string, id: string): Promise<Row> {
  if (!UUID_RE.test(id)) throw new HttpError(400, "invalid sample id");
  const row = await env.DB.prepare("SELECT * FROM samples WHERE id = ? AND project = ?")
    .bind(id, project)
    .first<Row>();
  if (!row) throw new HttpError(404, "sample not found");
  return row;
}

async function handlePresign(request: Request, env: Env, project: string): Promise<Response> {
  const text = await request.text();
  let body: unknown = {};
  if (text.trim() !== "") {
    try {
      body = JSON.parse(text);
    } catch {
      throw new HttpError(400, "body must be JSON");
    }
  }
  if (!isObj(body)) throw new HttpError(400, "body must be a JSON object");
  const meta = validateMeta(body);

  const existing = await env.DB.prepare("SELECT * FROM samples WHERE id = ?").bind(meta.id).first<Row>();
  let imageKey: string;
  let contentType: string;
  if (existing) {
    // Retry of an earlier presign (e.g. the URL expired while the device was offline).
    if (existing.project !== project) throw new HttpError(409, "id already used by another project");
    if (existing.uploaded) throw new HttpError(409, "sample already uploaded");
    imageKey = existing.image_key;
    contentType = existing.content_type ?? meta.content_type;
  } else {
    const receivedAt = new Date().toISOString();
    contentType = meta.content_type;
    imageKey = `${project}/${receivedAt.slice(0, 10)}/${meta.id}.${CONTENT_TYPES[contentType]}`;
    await env.DB.prepare(
      `INSERT INTO samples (id, project, image_key, content_type, device_id, model_version,
         image_width, image_height, bbox_format, captured_at, received_at, detections, tags)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        meta.id,
        project,
        imageKey,
        contentType,
        meta.device_id,
        meta.model_version,
        meta.image_width,
        meta.image_height,
        meta.bbox_format,
        meta.captured_at,
        receivedAt,
        meta.detections ? JSON.stringify(meta.detections) : null,
        meta.tags ? JSON.stringify(meta.tags) : null,
      )
      .run();
  }

  if (presignEnabled(env)) {
    return json({
      id: meta.id,
      image_key: imageKey,
      upload_mode: "presigned",
      upload_url: await presignPut(env, imageKey, contentType),
      upload_headers: { "content-type": contentType },
      expires_in: PRESIGN_TTL_SECONDS,
    });
  }
  const origin = new URL(request.url).origin;
  return json({
    id: meta.id,
    image_key: imageKey,
    upload_mode: "direct",
    upload_url: `${origin}/v1/${project}/upload/${meta.id}`,
    upload_headers: { "content-type": contentType },
  });
}

async function handleDirectUpload(request: Request, env: Env, project: string, id: string): Promise<Response> {
  const row = await getRow(env, project, id);
  if (row.uploaded) throw new HttpError(409, "sample already uploaded");
  if (Number(request.headers.get("content-length") ?? "0") > MAX_DIRECT_UPLOAD_BYTES)
    throw new HttpError(413, "image too large");
  // Buffer instead of streaming: R2 rejects streams of unknown length, and this enforces the real size.
  const data = await request.arrayBuffer();
  if (data.byteLength === 0) throw new HttpError(400, "empty body");
  if (data.byteLength > MAX_DIRECT_UPLOAD_BYTES) throw new HttpError(413, "image too large");
  await env.BUCKET.put(row.image_key, data, {
    httpMetadata: { contentType: row.content_type ?? "application/octet-stream" },
  });
  return json({ id, image_key: row.image_key });
}

async function handleConfirm(env: Env, project: string, id: string): Promise<Response> {
  const row = await getRow(env, project, id);
  const head = await env.BUCKET.head(row.image_key);
  if (!head) throw new HttpError(409, "object not found in bucket; upload it first");
  await env.DB.prepare("UPDATE samples SET uploaded = 1 WHERE id = ?").bind(id).run();
  return json({ id, uploaded: true, size: head.size });
}

async function handleProjects(env: Env): Promise<Response> {
  const { results } = await env.DB.prepare(
    `SELECT project, COUNT(*) AS total, SUM(uploaded) AS uploaded, SUM(archived) AS archived,
            MAX(received_at) AS last_received_at
     FROM samples GROUP BY project ORDER BY project`,
  ).all();
  return json({ projects: results });
}

function decodeRow(row: Row) {
  return {
    ...row,
    uploaded: Boolean(row.uploaded),
    archived: Boolean(row.archived),
    detections: row.detections ? JSON.parse(row.detections) : null,
    tags: row.tags ? JSON.parse(row.tags) : null,
  };
}

async function handleSamples(url: URL, env: Env, project: string): Promise<Response> {
  const limit = Math.min(Math.max(Number(url.searchParams.get("limit") ?? 500) || 500, 1), 1000);
  const since = url.searchParams.get("since");
  const cursor = url.searchParams.get("cursor");
  const uploadedOnly = url.searchParams.get("uploaded") !== "all";

  const where = ["project = ?"];
  const args: unknown[] = [project];
  if (uploadedOnly) where.push("uploaded = 1");
  if (since) {
    const t = Date.parse(since);
    if (Number.isNaN(t)) throw new HttpError(400, "since must be a date or ISO-8601 timestamp");
    where.push("received_at >= ?");
    args.push(new Date(t).toISOString());
  }
  if (cursor) {
    let cReceived: string, cId: string;
    try {
      [cReceived, cId] = JSON.parse(atob(cursor));
    } catch {
      throw new HttpError(400, "invalid cursor");
    }
    where.push("(received_at, id) > (?, ?)");
    args.push(cReceived, cId);
  }
  const { results } = await env.DB.prepare(
    `SELECT * FROM samples WHERE ${where.join(" AND ")} ORDER BY received_at, id LIMIT ?`,
  )
    .bind(...args, limit + 1)
    .all<Row>();

  const page = results.slice(0, limit);
  const last = page[page.length - 1] as Row | undefined;
  const nextCursor =
    results.length > limit && last ? btoa(JSON.stringify([last.received_at, last.id])) : null;
  return json({ samples: page.map(decodeRow), next_cursor: nextCursor });
}

async function handleGetImage(env: Env, project: string, id: string): Promise<Response> {
  const row = await getRow(env, project, id);
  const obj = await env.BUCKET.get(row.image_key);
  if (!obj) throw new HttpError(404, row.archived ? "image was archived (deleted from bucket)" : "image not found");
  return new Response(obj.body, {
    headers: {
      "content-type": obj.httpMetadata?.contentType ?? "application/octet-stream",
      "content-length": String(obj.size),
    },
  });
}

async function handleDeleteImage(env: Env, project: string, id: string): Promise<Response> {
  const row = await getRow(env, project, id);
  await env.BUCKET.delete(row.image_key);
  await env.DB.prepare("UPDATE samples SET archived = 1 WHERE id = ?").bind(id).run();
  return json({ id, archived: true });
}

async function route(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const parts = url.pathname.split("/").filter(Boolean);
  if (parts[0] !== "v1") throw new HttpError(404, "not found");
  await checkAuth(request, env);

  const method = request.method;
  if (parts.length === 2 && parts[1] === "projects" && method === "GET") return handleProjects(env);

  const project = parts[1];
  if (!project || !SLUG_RE.test(project))
    throw new HttpError(400, "project must match ^[a-z0-9][a-z0-9-]{1,62}$");
  const rest = parts.slice(2);

  if (rest.length === 1 && rest[0] === "presign" && method === "POST") return handlePresign(request, env, project);
  if (rest.length === 2 && rest[0] === "upload" && method === "PUT") return handleDirectUpload(request, env, project, rest[1]);
  if (rest.length === 2 && rest[0] === "confirm" && method === "POST") return handleConfirm(env, project, rest[1]);
  if (rest.length === 1 && rest[0] === "samples" && method === "GET") return handleSamples(url, env, project);
  if (rest.length === 3 && rest[0] === "samples" && rest[2] === "image") {
    if (method === "GET") return handleGetImage(env, project, rest[1]);
    if (method === "DELETE") return handleDeleteImage(env, project, rest[1]);
  }
  throw new HttpError(404, "not found");
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      return await route(request, env);
    } catch (err) {
      if (err instanceof HttpError) return json({ error: err.message }, err.status);
      console.error(err);
      return json({ error: "internal error" }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
