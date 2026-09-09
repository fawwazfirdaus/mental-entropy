# Empath Backend Context for MES Score Calculator Integration

This document provides context about the Empath backend to help the MES (Mental/Emotional State) score calculator integrate effectively.

---

## Overview

Empath is a therapist-client engagement platform with:
- **Backend**: Node.js/Express API server (port 4900)
- **Database**: MySQL with AES-256-CBC encryption
- **Vector DB**: Qdrant for semantic memory search
- **AI Services**: OpenAI, Anthropic Claude, Hume emotion detection

---

## Database Schema (Relevant Tables)

### Users
```sql
users (
  id UUID PRIMARY KEY,
  email VARCHAR UNIQUE,
  password_hash VARCHAR,
  name VARCHAR,
  IV VARCHAR,              -- Encryption initialization vector (per-user)
  deviceToken VARCHAR,
  created_at, updated_at
)
```

### Clients
```sql
clients (
  client_id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  phone_number VARCHAR,
  profile_picture_url VARCHAR,
  created_at, updated_at
)
```

### Journals (Primary Data Source for MES)
```sql
journals (
  journal_id UUID PRIMARY KEY,
  client_id UUID REFERENCES clients(client_id),
  entry_date DATETIME,
  encryptedJournal TEXT,   -- AES-256-CBC encrypted JSON (see structure below)
  isShared BOOLEAN,
  created_at, updated_at,
  embedded_at TIMESTAMP    -- When vector embedded
)
-- Indexes on: client_id, created_at, embedded_at
```

**Decrypted Journal Structure**:
```json
{
  "text": "Plain text journal content",
  "title": "AI-generated title",
  "mood_trigger": "Extracted trigger (e.g., 'work stress')",
  "feeling": -1.0 to 1.0,           // Valence (negative to positive)
  "intensity": 0.0 to 1.0,          // Arousal level
  "calculatedFeeling": -0.7 to 0.7, // Computed sentiment
  "calculatedIntensity": 0.0 to 0.9,// Computed arousal
  "approach_withdrawal": "approach" | "withdrawal",
  "platform": "ios" | "phone" | "web" | "sms" | "whatsapp",
  "voice_journal_path": "S3 URL (if voice entry)",
  "tintColor": "UI color code",
  "weatherIcon": "Weather at time of entry",
  "emotions": [                     // From Hume API
    { "name": "anxiety", "score": 0.72 },
    { "name": "sadness", "score": 0.45 }
  ]
}
```

### Mood (Standalone Mood Logs)
```sql
mood (
  id INT AUTO_INCREMENT PRIMARY KEY,
  client_id UUID,
  date DATE,
  encryptedFeeling TEXT,   -- Encrypted: [feeling, intensity]
  created_at
)
```

### Memories (Vector-Indexed Extracts)
```sql
memories (
  memory_id UUID PRIMARY KEY,
  source_id UUID,          -- journal_id or health_record_id
  memory_index INT,        -- Position in source
  client_id UUID,
  user_id UUID,
  encrypted_content TEXT,  -- AES-256-CBC encrypted memory text
  created_at
)
-- Paired with Qdrant vectors containing emotion/topic metadata
```

### Mentioned People (Relationship Tracking)
```sql
mentioned_people (
  person_id UUID PRIMARY KEY,
  client_id UUID,
  name VARCHAR,
  relationship_type VARCHAR,  -- 'family', 'friend', 'colleague', etc.
  created_at, updated_at
)

journal_mentions (
  mention_id UUID PRIMARY KEY,
  journal_id UUID,
  person_id UUID,
  context TEXT,              -- Snippet of how they were mentioned
  created_at
)
```

### Client-Therapist Relationships
```sql
client_therapist_links (
  link_id UUID PRIMARY KEY,
  client_id UUID,
  therapist_id UUID,
  relationship_status ENUM('active', 'inactive', 'ended'),
  assigned_at DATETIME,
  notes TEXT,
  UNIQUE(therapist_id, client_id)
)
```

---

## Encryption Details

**Algorithm**: AES-256-CBC
- **Key**: `process.env.SECRET_KEY` (32-byte hex string)
- **IV**: User-specific, stored in `users.IV` column
- **Usage**: All journal content, mood data, and memories are encrypted at rest

**Decryption Example** (Node.js):
```javascript
const crypto = require('crypto');

function decrypt(encryptedData, userIV) {
  const key = Buffer.from(process.env.SECRET_KEY, 'hex');
  const iv = Buffer.from(userIV, 'hex');
  const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
  let decrypted = decipher.update(encryptedData, 'base64', 'utf8');
  decrypted += decipher.final('utf8');
  return JSON.parse(decrypted);
}
```

---

## Relevant API Endpoints

### Authentication
All authenticated endpoints require `Authorization: Bearer <JWT>` header.

```
POST /api/users/login
Body: { email, password }
Returns: { accessToken, refreshToken, userId, clientId?, therapistId? }

POST /api/users/refreshLoginToken
Body: { refreshToken }
Returns: { accessToken }
```

### Journal Data Access

```
POST /api/journals/getJournalsForClient
Body: { clientId, limit?, offset? }
Returns: Array of journal objects (decrypted)

GET /api/journals/journals/:journalId
Returns: Single journal entry

POST /api/journals/generateJournalInsights
Body: { clientId, startDate, endDate }
Returns: { insights, themes, patterns, recommendations }
```

### Mood Data

```
POST /api/moods/chart-data
Body: { clientId, startDate, endDate }
Returns: { moodData: [{ date, feeling, intensity }] }

POST /api/moods/stability-analysis
Body: { clientId, startDate, endDate }
Returns: { stability, trends, volatility }
```

### Memory Search (Qdrant)

```
GET /api/journals/client/:clientId/memories/analytics
Returns: { totalMemories, byType, byEmotion, byTopic }

GET /api/journals/client/:clientId/memories/emotions/:emotions
Params: emotions = comma-separated (e.g., "anxiety,sadness")
Returns: Memories filtered by emotion

GET /api/journals/client/:clientId/memories/topics/:topics
Params: topics = comma-separated
Returns: Memories filtered by topic
```

### People Mentions

```
GET /api/journals/getAllMentionedPeople?clientId=:clientId
Returns: [{ person_id, name, relationship_type, mention_count }]

POST /api/journals/getPeopleWithJournalIds
Body: { clientId }
Returns: People with associated journal IDs
```

---

## Existing Analysis Features

### 1. Sentiment Analysis
- **Source**: Python Flask server (VADER) + keyword matching fallback
- **Output**: `feeling` (-1 to 1) and `intensity` (0 to 1)
- **Model**: Russell's 2D Emotion Circumplex

### 2. Emotion Detection (Hume API)
- **Method**: Streaming language emotion analysis
- **Output**: Top emotions with confidence scores
- **Filtering**: Excludes common journaling emotions (contemplation, realization)

### 3. Memory Extraction (Claude AI)
Each journal entry is processed to extract ~8 key memories:
```
Memory Types:
- FACTUAL_MEMORY: Facts about self, preferences, background
- EPISODIC_MEMORY: Specific events, experiences
- SEMANTIC_MEMORY: General beliefs, knowledge, values
- ACTION: Plans, intentions, commitments, goals
```

### 4. Mood Stability Calculation
- Variance analysis over time periods
- Trend detection (improving/declining)
- Pattern identification

### 5. Journal Insights (Claude/GPT)
- Thematic analysis across date ranges
- Trigger identification
- Behavioral pattern detection
- Therapy recommendations

---

## Data Flow for Journal Creation

```
1. User submits journal entry (text/voice)
         │
         ▼
2. Parallel Processing:
   ├─ Sentiment Analysis (Python/VADER)
   ├─ Title & Trigger Extraction (GPT)
   └─ Emotion Detection (Hume API)
         │
         ▼
3. Encrypt & Store in MySQL
         │
         ▼
4. Async Background Processing:
   ├─ Extract mentioned people
   ├─ Extract memories (Claude)
   ├─ Generate embeddings (OpenAI)
   └─ Upsert to Qdrant vector DB
```

---

## Integration Recommendations for MES Calculator

### Option 1: API Integration
Use the existing REST endpoints with JWT authentication:
- Fetch journals via `/api/journals/getJournalsForClient`
- Access mood data via `/api/moods/chart-data`
- Query memories via `/api/journals/client/:clientId/memories/*`

**Pros**: No database access needed, uses existing auth
**Cons**: Network overhead, rate limits

### Option 2: Direct Database Access
Connect directly to MySQL with shared credentials:
```javascript
const mysql = require('mysql');
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME
});
```

**Pros**: Full data access, no API limits
**Cons**: Needs encryption key, tighter coupling

### Option 3: Shared Service Layer
Create a shared npm package with:
- Database queries
- Decryption utilities
- Data transformation helpers

**Pros**: Reusable, consistent data access
**Cons**: Maintenance overhead

---

## Key Data Points for MES Scoring

From each journal entry, you can extract:

| Field | Type | Description |
|-------|------|-------------|
| `feeling` | float (-1 to 1) | Emotional valence |
| `intensity` | float (0 to 1) | Emotional arousal |
| `emotions[]` | array | Hume-detected emotions with scores |
| `approach_withdrawal` | string | Behavioral tendency |
| `mood_trigger` | string | Identified trigger |
| `text` | string | Raw journal content |
| `entry_date` | datetime | When entry was made |
| `platform` | string | Entry source (ios/web/phone/sms) |

From mood table:
| Field | Type | Description |
|-------|------|-------------|
| `feeling` | float | Standalone mood valence |
| `intensity` | float | Standalone mood arousal |
| `date` | date | Date of mood log |

From memories (Qdrant):
| Field | Type | Description |
|-------|------|-------------|
| `memory_type` | string | FACTUAL/EPISODIC/SEMANTIC/ACTION |
| `emotions[]` | array | Emotions in this memory |
| `topics[]` | array | Key concepts/people/places |

---

## Environment Variables (Relevant)

```env
# Database
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_NAME=

# Encryption
SECRET_KEY=              # 32-byte hex for AES-256

# Vector DB
QDRANT_URL=
QDRANT_API_KEY=

# AI Services (if MES needs its own analysis)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUME_API_KEY=

# Auth
JSON_WEB_TOKEN_SECRET=
REFRESH_TOKEN_SECRET=
```

---

## File Structure Reference

```
backend/
├── server.js                          # Express entry point
├── db.js                              # MySQL connection pool
├── controllers/
│   ├── journals/
│   │   ├── journalsController.js      # Journal CRUD
│   │   ├── journalSummaryController.js
│   │   ├── journalInsightsController.js
│   │   └── memoriesController.js
│   ├── moods/
│   │   └── moodAnalysis.js            # Mood calculations
│   └── clients/
│       └── clientsController.js
├── services/
│   ├── journalCreationService.js      # Journal processing
│   ├── journalEmbeddingService.js     # Memory extraction
│   ├── humeEmotionService.js          # Emotion detection
│   ├── vectorStore.js                 # Qdrant wrapper
│   └── ragService.js                  # RAG implementation
├── middleware/Auth/
│   ├── verifyToken.js                 # JWT middleware
│   └── Encryption/encrypt.js          # Encryption utilities
└── routes/
    ├── journals.js
    ├── clients.js
    ├── moods.js
    └── ...
```

---

## Notes for MES Implementation

1. **Longitudinal Analysis**: Journals have `entry_date` and `created_at` - use `entry_date` for user-intended date, `created_at` for actual submission time.

2. **Multi-Therapist Support**: Clients can have multiple therapists via `client_therapist_links`. Consider whose view the MES score is for.

3. **Platform Diversity**: Journal entries come from various sources (iOS app, web, phone calls, SMS). Platform affects entry length/quality.

4. **Encryption Requirement**: All PII and journal content is encrypted. MES repo will need access to `SECRET_KEY` for direct DB access.

5. **Existing Emotion Data**: Hume emotions are already extracted - consider using these rather than re-analyzing.

6. **Memory Types**: The 4 memory types (Factual/Episodic/Semantic/Action) provide structured insight into journal content.

7. **Rate Limiting**: If using APIs, be aware of potential rate limits on insight generation endpoints.

---

## Contact

For database credentials, `.env` files, or questions:
- Karan Patil
- Div Khare
- Discord channel (see README.md)
