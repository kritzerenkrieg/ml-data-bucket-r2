CREATE TABLE IF NOT EXISTS samples (
  id TEXT PRIMARY KEY,            -- uuid (client-generated so retries are idempotent)
  project TEXT NOT NULL,
  image_key TEXT NOT NULL,        -- R2 key: {project}/{yyyy-mm-dd}/{id}.{ext}
  content_type TEXT,
  device_id TEXT,
  model_version TEXT,
  image_width INTEGER,
  image_height INTEGER,
  bbox_format TEXT,               -- yolo | coco | xyxy | NULL
  captured_at TEXT,
  received_at TEXT NOT NULL,
  uploaded INTEGER NOT NULL DEFAULT 0,  -- 1 once the object is confirmed in R2
  archived INTEGER NOT NULL DEFAULT 0,  -- 1 once the object was deleted from R2 after export
  detections TEXT,                -- raw JSON array or NULL
  tags TEXT                       -- raw JSON object or NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_project_time ON samples(project, received_at, id);
