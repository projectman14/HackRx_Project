# Supabase Setup Guide for HackRx 6.0

This guide will help you set up Supabase for the HackRx advanced model using the `supabase-py` client.

## Prerequisites

1. **Supabase Account**: Sign up at [supabase.com](https://supabase.com)
2. **Python Environment**: Python 3.8+ with pip
3. **Required Packages**: Already included in `requirements_advanced.txt`

## Step 1: Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign in
2. Click "New Project"
3. Choose your organization
4. Enter project details:
   - **Name**: `hackrx-advanced` (or your preferred name)
   - **Database Password**: Create a strong password
   - **Region**: Choose the closest region to you
5. Click "Create new project"
6. Wait for the project to be created (usually 1-2 minutes)

## Step 2: Get Your Supabase Credentials

1. In your Supabase project dashboard, go to **Settings** → **API**
2. Copy the following values:
   - **Project URL**: Found in the "Project URL" field
   - **Anon/Public Key**: Found in the "anon public" field

## Step 3: Create Required Tables

1. In your Supabase dashboard, go to **SQL Editor**
2. Run the following SQL commands to create the required tables:

```sql
-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    url TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON chunks(embedding_id);
```

## Step 4: Configure Environment Variables

1. Create or update your `.env` file with the following variables:

```env
# Required
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your_anon_public_key_here
GOOGLE_API_KEY=your_google_api_key_here
API_KEY=hackrx_2024_secure_key_123

# Optional (for full functionality)
PINECONE_API_KEY=your_pinecone_api_key_here
```

**Important**: Replace `your-project-ref` and `your_anon_public_key_here` with your actual values from Supabase.

## Step 5: Test the Connection

Run the integration test to verify everything is working:

```bash
# On Windows
test_supabase_integration.bat

# On Linux/Mac
python test_supabase_integration.py
```

You should see output like:
```
🚀 HackRx 6.0 - Supabase Integration Test
==================================================
🧪 Testing Supabase Integration...
✅ Supabase connection successful
✅ Document insert successful
✅ Document select successful
✅ Document update successful
✅ Document delete successful

🧪 Testing Required Tables...
✅ Documents table exists
✅ Chunks table exists

✅ All tests passed! Supabase integration is ready.
```

## Step 6: Run the Advanced Model

Once the integration test passes, you can run the advanced model:

```bash
python main_advanced.py
```

## Troubleshooting

### Connection Issues

1. **"SUPABASE_URL or SUPABASE_KEY not found"**
   - Make sure your `.env` file exists and contains the correct credentials
   - Verify the URL format: `https://your-project-ref.supabase.co`
   - Check that you're using the anon/public key, not the service role key

2. **"Table not found" errors**
   - Ensure you've created the required tables in the SQL Editor
   - Check that the table names match exactly: `documents` and `chunks`

3. **"RPC function not found"**
   - This is expected if the `exec_sql` RPC function doesn't exist
   - Tables should be created manually in the SQL Editor

### Common Issues

1. **"Supabase initialization failed"**
   - Verify your project URL and API key are correct
   - Check that your Supabase project is active
   - Ensure you're using the anon/public key, not the service role key

2. **"Table creation via RPC failed"**
   - This is expected behavior if the `exec_sql` RPC function is not available
   - Create tables manually in the Supabase SQL Editor

3. **"Document insert failed"**
   - Check that the `documents` table exists
   - Verify the table schema matches the expected structure

## Security Best Practices

1. **Never commit your `.env` file** to version control
2. **Use the anon/public key** for client-side operations
3. **Enable Row Level Security (RLS)** in Supabase for production use
4. **Regularly rotate your API keys**

## Performance Tips

1. **Indexes**: The required indexes are created automatically for better query performance
2. **Connection Pooling**: Supabase handles connection pooling automatically
3. **Retry Logic**: The system includes retry logic for transient failures

## API Usage

The system now uses Supabase's REST API for all database operations:

- **Insert**: `supabase.table("documents").insert(data).execute()`
- **Select**: `supabase.table("documents").select("*").eq("id", "value").execute()`
- **Update**: `supabase.table("documents").update(data).eq("id", "value").execute()`
- **Delete**: `supabase.table("documents").delete().eq("id", "value").execute()`

## Next Steps

After successful setup:

1. Test the API endpoint with your accuracy test
2. Monitor the database usage in Supabase dashboard
3. Consider setting up monitoring and alerts
4. Scale your Supabase plan if needed

## Support

- **Supabase Documentation**: [supabase.com/docs](https://supabase.com/docs)
- **Supabase Community**: [github.com/supabase/supabase/discussions](https://github.com/supabase/supabase/discussions)
- **supabase-py Documentation**: [supabase.com/docs/reference/python](https://supabase.com/docs/reference/python)

## Migration from Direct PostgreSQL

If you're migrating from direct PostgreSQL connections:

1. Update your environment variables to use `SUPABASE_URL` and `SUPABASE_KEY`
2. Create the required tables in Supabase SQL Editor
3. Run the integration test to verify the setup
4. The application will automatically use Supabase's REST API

The code has been updated to use `supabase-py` instead of direct PostgreSQL connections, providing better integration with Supabase's features.
