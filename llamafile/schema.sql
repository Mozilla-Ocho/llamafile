CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model TEXT,
    title TEXT
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chat_id INTEGER,
    role TEXT,
    content TEXT,
    temperature REAL,
    top_p REAL,
    presence_penalty REAL,
    frequency_penalty REAL,
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
