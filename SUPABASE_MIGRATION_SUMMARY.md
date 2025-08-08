# Supabase Migration Summary

This document summarizes the changes made to migrate HackRx 6.0 from direct PostgreSQL connections to using the `supabase-py` client.

## Overview

The code has been updated to use Supabase's REST API instead of direct PostgreSQL connections, providing better integration with Supabase's features and improved security.

## Key Changes Made

### 1. Dependencies Updated

**Removed:**
- `psycopg2-binary==2.9.9` - Direct PostgreSQL connection library

**Kept:**
- `supabase` - Supabase Python client (already present)

### 2. Database Connection Changes

**Before:**
```python
# PostgreSQL imports
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json

# Global state
db_pool = None
```

**After:**
```python
from supabase import create_client, Client

# Global state
supabase_client: Optional[Client] = None
```

### 3. Initialization Function Changes

**Before:**
```python
def initialize_postgresql_supabase(max_retries=5, retry_delay=3):
    """Initialize Supabase client and create tables if not exist with retries."""
    # Direct PostgreSQL connection logic
    # Connection pooling setup
    # Raw SQL execution via RPC
```

**After:**
```python
def initialize_supabase():
    """Initialize Supabase client and create tables if not exist."""
    # Supabase client initialization
    # REST API connection test
    # Table creation via RPC (if available)

def create_supabase_tables():
    """Create tables in Supabase if they don't exist."""
    # Separate table creation function
    # Better error handling for RPC functions
```

### 4. Database Operations Changes

**Before (Direct PostgreSQL):**
```python
def store_document_in_db(document_id: str, url: str, content: str, chunks: List[Dict]):
    """Store document and chunks in PostgreSQL"""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO documents (id, url, content_hash, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    url = EXCLUDED.url,
                    content_hash = EXCLUDED.content_hash,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP
            """, (document_id, url, content_hash, Json({"total_chunks": len(chunks)})))
        conn.commit()
    finally:
        db_pool.putconn(conn)
```

**After (Supabase REST API):**
```python
def store_document_in_db(document_id: str, url: str, content: str, chunks: List[Dict]):
    """Store document and chunks in Supabase"""
    try:
        # Store document with upsert
        supabase_client.table("documents").upsert({
            "id": document_id,
            "url": url,
            "content_hash": content_hash,
            "metadata": {"total_chunks": len(chunks)}
        }).execute()
        
        # Store chunks with upsert
        for chunk in chunks:
            supabase_client.table("chunks").upsert({
                "document_id": document_id,
                "chunk_index": chunk["index"],
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            }).execute()
    except Exception as e:
        print(f"❌ Database storage failed: {e}")
        return False
```

### 5. Search Operations Changes

**Before (Direct SQL):**
```python
# PostgreSQL keyword search
conn = db_pool.getconn()
try:
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        term_conditions = " OR ".join([f"content ILIKE %s" for _ in key_terms])
        query = f"""
            SELECT content, chunk_index, metadata
            FROM chunks 
            WHERE document_id = %s AND ({term_conditions})
            ORDER BY chunk_index
            LIMIT {top_k}
        """
        params = [document_id] + [f"%{term}%" for term in key_terms]
        cursor.execute(query, params)
```

**After (Supabase REST API):**
```python
# Supabase keyword search
search_conditions = []
for term in key_terms:
    search_conditions.append(f"content.ilike.%{term}%")

response = supabase_client.table("chunks").select(
    "content, chunk_index, metadata"
).eq("document_id", document_id).or_(
    ",".join(search_conditions)
).order("chunk_index").limit(top_k).execute()
```

### 6. Environment Variables Changes

**Before:**
```env
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres
```

**After:**
```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your_anon_public_key_here
```

### 7. Error Handling Improvements

- Better error messages for Supabase-specific issues
- Fallback mechanisms for when RPC functions are not available
- Graceful handling of table creation failures
- Improved retry logic for transient failures

### 8. New Test Files

**Added:**
- `test_supabase_integration.py` - Comprehensive Supabase integration test
- `test_supabase_integration.bat` - Windows batch file for testing

## Benefits of the Migration

### 1. **Better Security**
- Uses Supabase's built-in security features
- No direct database credentials in connection strings
- Automatic SSL/TLS encryption

### 2. **Improved Performance**
- Supabase handles connection pooling automatically
- Optimized REST API calls
- Better caching mechanisms

### 3. **Enhanced Features**
- Real-time subscriptions (if needed)
- Built-in authentication
- Row Level Security (RLS) support
- Automatic backups

### 4. **Easier Maintenance**
- No need to manage database connections manually
- Automatic retry logic
- Better error handling
- Simplified deployment

### 5. **Better Integration**
- Native Supabase features
- REST API instead of raw SQL
- Type-safe operations
- Better debugging capabilities

## Migration Steps for Users

1. **Update Environment Variables**
   - Replace `DATABASE_URL` with `SUPABASE_URL` and `SUPABASE_KEY`
   - Get credentials from Supabase dashboard

2. **Create Tables**
   - Run the provided SQL in Supabase SQL Editor
   - Or use the automatic table creation (if RPC available)

3. **Test Integration**
   - Run `test_supabase_integration.bat` (Windows)
   - Or `python test_supabase_integration.py` (Linux/Mac)

4. **Update Documentation**
   - Follow the updated `SUPABASE_SETUP_GUIDE.md`
   - Use the new test files for verification

## Backward Compatibility

The migration maintains backward compatibility for:
- API endpoints and responses
- Document processing logic
- Vector database operations (Pinecone)
- Google AI integration

Only the database layer has been changed, so existing functionality remains the same.

## Troubleshooting

### Common Issues

1. **"SUPABASE_URL or SUPABASE_KEY not found"**
   - Check your `.env` file
   - Verify the credentials from Supabase dashboard

2. **"Table not found" errors**
   - Create tables manually in Supabase SQL Editor
   - Check table names and schema

3. **"RPC function not found"**
   - Expected if `exec_sql` RPC is not available
   - Create tables manually

### Support

- Updated `SUPABASE_SETUP_GUIDE.md` with new instructions
- New test files for verification
- Comprehensive error handling and logging

## Conclusion

The migration to `supabase-py` provides a more robust, secure, and maintainable solution for HackRx 6.0. The changes improve the overall architecture while maintaining all existing functionality.
