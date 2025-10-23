import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import sqlglot
from sqlglot import parse_one, transpile
from sqlglot.errors import ParseError

# --------------------- MODEL + DB ---------------------

MODEL_ID = "defog/sqlcoder-7b-2"

def get_db():
    return mysql.connector.connect(
        host="68.178.155.255",
        user="Anika12",
        password="Anika12",
        database="ems",
        autocommit=True,
        pool_name="mypool",
        pool_size=3,
        pool_reset_session=True
    )

def ensure_connection(conn):
    """Ensure the connection is alive, reconnect if needed."""
    try:
        conn.ping(reconnect=True, attempts=3, delay=1)
        return conn
    except mysql.connector.Error:
        # Reconnect if ping fails
        return get_db()

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True,
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    mdl.eval()
    return tok, mdl

# --------------------- PROMPT ---------------------

SCHEMA_AND_RULES = """
### Context
You are an expert at writing SQL SELECT queries.

We have exactly **two tables** and may only use these columns:

Table `leavebalance`
- id (int, PK, auto-increment): row id
- empname (varchar): employee name
- empemail (varchar): employee email
- cl (varchar): remaining casual leave count (numeric text)
- sl (varchar): remaining sick leave count (numeric text)
- co (varchar): remaining comp-off count (numeric text)
- lastupdate (datetime): when this balance row was last updated
- icl (varchar): total allocated casual leaves till now
- isl (varchar): total allocated sick leaves till now
- ico (varchar): total allocated comp-off leaves till now
- iupdate (datetime): when allocation was last updated

Meaning:
- **Leave balance (total remaining)** = COALESCE(cl,0)+COALESCE(sl,0)+COALESCE(co,0)
- When asking for current/most recent balance per employee, choose the **latest row** by:
  1) prefer the row with the greatest non-NULL `lastupdate`; if `lastupdate` is NULL, then
  2) choose the row with the **highest `id`**.

Table `leaves`
- ID (int, PK, auto-increment): leave application id
- empname (varchar): employee name
- empemail (varchar): employee email
- leavetype (varchar): 'SICK LEAVE' | 'CASUAL LEAVE' | 'COMP OFF'
- applied (timestamp): when the leave was applied
- `from` (datetime): leave start
- `to` (datetime): leave end
- desg (varchar): designation
- reason (varchar): reason text
- empph (varchar): **phone number**
- work_location (varchar): place of work

Extra meanings:
- **Phone numbers** are only stored in `leaves.empph` (not in leavebalance).
- **Eligibility to apply for leave** can be derived as (leave balance total > 0), unless the question asks about a specific type (then check that specific remaining bucket).
- Time filters use datetime ranges.
- Always produce **one** SQL **SELECT** statement ending with a semicolon.
- Use explicit column names (avoid SELECT *).
- If the question cannot be answered strictly from these two tables/columns, output exactly: __SORRY__
- Never use other tables.
"""

def build_prompt(user_question: str) -> str:
    return f"{SCHEMA_AND_RULES}\n### Question:\n{user_question}\n\n### SQL (one SELECT or __SORRY__):\n"

# --------------------- SQL TRANSPILER (INDUSTRY LEVEL) ---------------------

def transpile_to_mysql(sql: str) -> str:
    """
    Use sqlglot to automatically transpile any SQL dialect to MySQL.
    This handles PostgreSQL, SQLite, MSSQL, Oracle syntax automatically.
    """
    sql = sql.strip()
    
    # Basic validation
    if not re.match(r"^\s*SELECT\b", sql, flags=re.IGNORECASE):
        return "__SORRY__"
    
    # Remove code fences
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    
    try:
        # Try to parse and transpile to MySQL
        # sqlglot will auto-detect the source dialect
        mysql_sql = transpile(sql, read=None, write="mysql")[0]
        
        # Ensure semicolon
        if not mysql_sql.rstrip().endswith(";"):
            mysql_sql = mysql_sql.rstrip() + ";"
            
        return mysql_sql
        
    except ParseError as e:
        # If parsing fails, log and return error
        print(f"⚠️  SQL Parse Error: {e}")
        return "__SORRY__"
    except Exception as e:
        # For any other errors, try to return original with basic cleanup
        print(f"⚠️  Transpilation Error: {e}")
        if not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"
        return sql

def validate_mysql_syntax(conn, sql: str) -> tuple[bool, str]:
    """
    Validate SQL by using MySQL's EXPLAIN without executing.
    Returns (is_valid, error_message)
    """
    if sql.strip() == "__SORRY__":
        return False, "Cannot answer with available schema"
    
    cursor = None
    try:
        # Ensure connection is alive
        conn.ping(reconnect=True, attempts=2, delay=0.5)
        
        cursor = conn.cursor()
        # Use EXPLAIN to validate without executing
        # Remove semicolon for EXPLAIN
        sql_to_validate = sql.rstrip(';').strip()
        cursor.execute(f"EXPLAIN {sql_to_validate}")
        cursor.fetchall()
        return True, ""
        
    except mysql.connector.Error as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
    finally:
        if cursor:
            cursor.close()

# --------------------- GENERATE + EXECUTE ---------------------

def generate_sql(tokenizer, model, prompt: str, max_new_tokens: int = 220) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Extract after our marker
    m = re.search(r"### SQL.*?:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    sql_block = m.group(1).strip() if m else text.strip()

    # Take until first semicolon (inclusive)
    semi = sql_block.find(";")
    raw_sql = (sql_block[:semi+1] if semi != -1 else sql_block).strip()

    if "__SORRY__" in raw_sql:
        return "__SORRY__"

    return raw_sql

def run_query(conn, sql: str):
    if sql.strip() == "__SORRY__":
        return "__SORRY__: I can only answer using the allowed columns of `leavebalance` and `leaves`."
    
    cursor = None
    try:
        # Ensure connection is alive
        conn.ping(reconnect=True, attempts=2, delay=0.5)
        
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return cols, rows
        
    except mysql.connector.Error as e:
        raise Exception(f"Query execution failed: {e}")
    finally:
        if cursor:
            cursor.close()

# --------------------- CLI WITH AUTO-CORRECTION ---------------------

def main():
    print("🔧 Loading model...")
    tokenizer, model = load_model()
    print("✅ Model ready")
    conn = get_db()
    print("✅ DB connected\n")

    print("Ask about only `leavebalance` and `leaves`. Type 'quit' to exit.")
    
    while True:
        try:
            q = input("\n💬 Question: ").strip()
            if q.lower() == "quit":
                break
            
            # Ensure connection is alive before processing
            conn = ensure_connection(conn)
            
            # Generate SQL from model
            prompt = build_prompt(q)
            raw_sql = generate_sql(tokenizer, model, prompt)
            print(f"\n📝 Generated SQL:\n{raw_sql}")
            
            if raw_sql == "__SORRY__":
                print("❌ Cannot answer with available schema")
                continue
            
            # Transpile to MySQL
            mysql_sql = transpile_to_mysql(raw_sql)
            print(f"\n📜 MySQL SQL:\n{mysql_sql}")
            
            if mysql_sql == "__SORRY__":
                print("❌ SQL parsing failed")
                continue
            
            # Validate before executing
            is_valid, error_msg = validate_mysql_syntax(conn, mysql_sql)
            if not is_valid:
                print(f"⚠️  SQL Validation Failed: {error_msg}")
                print("Attempting to execute anyway...")
                # Don't skip - sometimes EXPLAIN fails but query works
            
            # Execute
            result = run_query(conn, mysql_sql)
            print("📊 Result:", result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            # Try to reconnect
            try:
                conn = get_db()
                print("♻️  Reconnected to database")
            except:
                print("❌ Failed to reconnect. Please restart.")
                break

    try:
        conn.close()
    except:
        pass
    print("👋 Bye")

if __name__ == "__main__":
    main()
