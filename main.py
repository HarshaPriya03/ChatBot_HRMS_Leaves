import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import datetime
from sentence_transformers import SentenceTransformer, util

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
    try:
        conn.ping(reconnect=True, attempts=3, delay=1)
        return conn
    except mysql.connector.Error:
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

# --------------------- DATE PREPROCESSING ---------------------

def preprocess_question(question: str) -> tuple:
    q_lower = question.lower()
    today = datetime.date.today()
    date_context = ""
    
    if "last month" in q_lower:
        last_month_start = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
        last_month_end = today.replace(day=1) - datetime.timedelta(days=1)
        date_context = f"Date range: {last_month_start} to {last_month_end}"
        question = question.replace("last month", f"between '{last_month_start}' and '{last_month_end}'")
    
    elif "last 30 days" in q_lower or "past 30 days" in q_lower:
        date_30_days_ago = today - datetime.timedelta(days=30)
        date_context = f"Date range: {date_30_days_ago} to {today}"
        question = question.replace("last 30 days", f"after '{date_30_days_ago}'")
        question = question.replace("past 30 days", f"after '{date_30_days_ago}'")
    
    elif "this month" in q_lower:
        month_start = today.replace(day=1)
        date_context = f"Date range: {month_start} to {today}"
        question = question.replace("this month", f"after '{month_start}'")
    
    elif "yesterday" in q_lower:
        yesterday = today - datetime.timedelta(days=1)
        date_context = f"Date: {yesterday}"
        question = question.replace("yesterday", f"on '{yesterday}'")
    
    elif "last week" in q_lower:
        last_week_start = today - datetime.timedelta(days=today.weekday() + 7)
        last_week_end = last_week_start + datetime.timedelta(days=6)
        date_context = f"Date range: {last_week_start} to {last_week_end}"
        question = question.replace("last week", f"between '{last_week_start}' and '{last_week_end}'")
    
    return question, date_context

# --------------------- INTENT DETECTION ---------------------

class IntentDetector:
    def __init__(self):
        print("🔧 Loading intent detector...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.intent_examples = {
            "leavebalance": [
                "remaining leaves", "leave balance", "how many leaves left",
                "available leaves", "leaves available", "pending leave count",
                "current balance", "check balance", "balance remaining",
                "CL left", "SL available", "comp off balance",
                "can I take leave", "eligible for leave", "eligibility",
                "balance of", "leavebalance of", "what is balance"
            ],
            "leaves": [
                "applied leaves", "leave history", "leaves taken", 
                "reason for leave", "leave from date to date",
                "when did apply", "past leaves", "leave applications",
                "how many days applied", "total days taken",
                "leaves in January", "on leave today", "who is on leave",
                "leave between dates", "applied on", "show leaves",
                "latest leave", "last leave", "recent leave", "when applied",
                "who applied", "which employee", "person applied", "members on leave"
            ]
        }
        
        self.intent_embeds = {
            k: self.model.encode(v, convert_to_tensor=True) 
            for k, v in self.intent_examples.items()
        }
        print("✅ Intent detector ready")
    
    def detect(self, question: str):
        q_emb = self.model.encode(question, convert_to_tensor=True)
        scores = {
            intent: float(util.cos_sim(q_emb, emb).max()) 
            for intent, emb in self.intent_embeds.items()
        }
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        if best_score < 0.35:
            q_lower = question.lower()
            if any(word in q_lower for word in ['balance', 'remaining', 'left', 'available', 'eligible']):
                return "leavebalance", 0.6
            elif any(word in q_lower for word in ['applied', 'history', 'latest', 'reason', 'when', 'days', 'who']):
                return "leaves", 0.6
            return "unknown", best_score
        
        return best_intent, best_score

# --------------------- CONTEXT ANALYSIS ---------------------

def analyze_query_context(question: str) -> dict:
    """
    Analyzes the question to extract context without hardcoding keywords.
    Returns context that helps generate better SQL.
    """
    q_lower = question.lower()
    context = {
        'has_specific_email': False,
        'email': None,
        'is_general_query': False,
        'query_scope': 'unknown'
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, question)
    if email_match:
        context['has_specific_email'] = True
        context['email'] = email_match.group()
        context['query_scope'] = 'specific_employee'
        return context
    
    # Detect if asking about multiple/all employees
    # Look for question words that typically ask about multiple people
    question_indicators = ['who', 'which', 'what', 'how many']
    person_plurals = ['employees', 'members', 'people', 'persons', 'staff']
    quantifiers = ['all', 'any', 'every', 'everyone', 'anyone']
    
    has_question_word = any(word in q_lower for word in question_indicators)
    has_plural = any(word in q_lower for word in person_plurals)
    has_quantifier = any(word in q_lower for word in quantifiers)
    
    # If asking "who/which/what" OR mentions plural people OR uses quantifiers
    if has_question_word or has_plural or has_quantifier:
        context['is_general_query'] = True
        context['query_scope'] = 'all_employees'
    else:
        # Default to specific if unclear (safer for privacy)
        context['query_scope'] = 'unclear'
    
    return context

# --------------------- SCHEMA TEMPLATES ---------------------

SCHEMA_TEMPLATES = {
    "leavebalance": """
Table: leavebalance
Columns:
  - id: INT PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email)
  - cl: VARCHAR (casual leave remaining)
  - sl: VARCHAR (sick leave remaining)
  - co: VARCHAR (comp-off remaining)
  - lastupdate: DATETIME
  - icl, isl, ico: VARCHAR (initial allocations)
  - iupdate: DATETIME

Rules:
- Column is 'empemail' NOT 'email'
- Calculate total: CAST(COALESCE(cl,'0') AS DECIMAL) + CAST(COALESCE(sl,'0') AS DECIMAL) + CAST(COALESCE(co,'0') AS DECIMAL)
- For latest: ORDER BY COALESCE(lastupdate, '1900-01-01') DESC, id DESC LIMIT 1
""",
    
    "leaves": """
Table: leaves
Columns:
  - ID: INT PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email)
  - leavetype: VARCHAR ('SICK LEAVE', 'CASUAL LEAVE', 'COMP OFF')
  - applied: TIMESTAMP
  - `from`: DATETIME (MUST use backticks - reserved keyword)
  - `to`: DATETIME (MUST use backticks - reserved keyword)
  - desg: VARCHAR (designation)
  - reason: VARCHAR
  - empph: VARCHAR
  - work_location: VARCHAR

Rules:
- Column is 'empemail' NOT 'email'
- MUST use backticks: `from` and `to`
- Days: DATEDIFF(`to`, `from`) + 1
- For latest: ORDER BY applied DESC LIMIT 1
"""
}

def get_schema_for_intent(intent: str) -> str:
    if intent == "leavebalance":
        return SCHEMA_TEMPLATES["leavebalance"]
    elif intent == "leaves":
        return SCHEMA_TEMPLATES["leaves"]
    else:
        return SCHEMA_TEMPLATES["leavebalance"] + "\n" + SCHEMA_TEMPLATES["leaves"]

# --------------------- PROMPT BUILDING ---------------------

def build_prompt(user_question: str, intent: str, context: dict) -> str:
    schema = get_schema_for_intent(intent)
    
    # Build context hints dynamically based on analysis
    context_hint = ""
    if context['has_specific_email']:
        context_hint = f"\nIMPORTANT: Filter by WHERE empemail = '{context['email']}'\n"
    elif context['is_general_query']:
        context_hint = "\nIMPORTANT: Query asks about multiple/all employees. DO NOT filter by specific email. Include empemail in SELECT.\n"
    
    prompt = f"""### Task
Generate a valid MySQL/MariaDB SELECT query.

### Schema
{schema}

### MySQL/MariaDB Requirements
- NO PostgreSQL syntax (NULLS FIRST/LAST, ILIKE, INTERVAL 'text', etc.)
- Date arithmetic: DATE_SUB(CURDATE(), INTERVAL X DAY/MONTH/YEAR)
- Reserved words need backticks: `from`, `to`
- Use LIKE for pattern matching, not ILIKE
{context_hint}

### Question
{user_question}

### Output only the SQL query with semicolon:
"""
    return prompt

# --------------------- LLM-BASED SQL REPAIR ---------------------

def repair_sql_with_llm(sql: str, error_msg: str, tokenizer, model, schema: str, original_question: str) -> str:
    """
    Uses the LLM to repair broken SQL by understanding the error.
    This is much more powerful than regex-based fixes.
    """
    
    repair_prompt = f"""### Task
Fix this SQL query to be valid MySQL/MariaDB syntax. The query failed with an error.

### Database Schema
{schema}

### Original Question
{original_question}

### Failed SQL Query
{sql}

### Error Message
{error_msg}

### Common MySQL/MariaDB Issues to Fix
1. Remove NULLS FIRST/LAST (not supported)
2. Convert INTERVAL 'X unit' → INTERVAL X UNIT
3. Convert CURRENT_DATE - INTERVAL → DATE_SUB(CURDATE(), INTERVAL ...)
4. Replace ILIKE with LIKE
5. Add backticks for reserved words: `from`, `to`
6. Fix syntax errors
7. Preserve table and column names exactly as in schema
8. Keep the semantic meaning of the original query

### Instructions
- Analyze the error message
- Fix ONLY the syntax issues
- Do NOT change the query logic or intent
- Output ONLY the corrected SQL query with semicolon
- No explanations, just the fixed query

### Corrected SQL:
"""
    
    inputs = tokenizer(repair_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Extract the corrected SQL
    if "### Corrected SQL:" in text:
        fixed_sql = text.split("### Corrected SQL:")[-1].strip()
    else:
        fixed_sql = text.replace(repair_prompt, "").strip()
    
    # Clean up
    fixed_sql = re.sub(r"^```sql\s*|\s*```$", "", fixed_sql, flags=re.IGNORECASE).strip()
    
    if not fixed_sql.endswith(';'):
        fixed_sql = fixed_sql.rstrip() + ';'
    
    return fixed_sql

# --------------------- SQL GENERATION ---------------------

def generate_sql(tokenizer, model, prompt: str, max_new_tokens: int = 350) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    if "### Output only the SQL query with semicolon:" in text:
        sql = text.split("### Output only the SQL query with semicolon:")[-1].strip()
    elif "### SQL:" in text:
        sql = text.split("### SQL:")[-1].strip()
    else:
        sql = text.replace(prompt, "").strip()
    
    # Clean up
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    
    # Take first complete statement
    if ';' in sql:
        sql = sql.split(';')[0].strip() + ';'
    elif not sql.endswith(';'):
        sql += ';'
    
    if "__SORRY__" in sql.upper() or "cannot" in sql.lower():
        return "__SORRY__"
    
    return sql

def validate_and_repair_sql(conn, sql: str, tokenizer, model, schema: str, original_question: str, max_attempts: int = 2) -> tuple:
    """
    Validates SQL and uses LLM to repair if invalid.
    Returns (final_sql, is_valid, attempt_count)
    """
    
    current_sql = sql
    
    for attempt in range(max_attempts):
        # Try to validate
        cursor = None
        try:
            conn.ping(reconnect=True, attempts=2, delay=0.5)
            cursor = conn.cursor()
            sql_to_validate = current_sql.rstrip(';').strip()
            cursor.execute(f"EXPLAIN {sql_to_validate}")
            cursor.fetchall()
            # Success!
            return current_sql, True, attempt + 1
        except mysql.connector.Error as e:
            error_msg = str(e)
            print(f"\n⚠️  Validation error (attempt {attempt + 1}/{max_attempts}): {error_msg}")
            
            if attempt < max_attempts - 1:
                print("🔧 Using LLM to repair SQL...")
                current_sql = repair_sql_with_llm(current_sql, error_msg, tokenizer, model, schema, original_question)
                print(f"🔄 Repaired SQL:\n{current_sql}\n")
            else:
                return current_sql, False, attempt + 1
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n⚠️  Unexpected error (attempt {attempt + 1}/{max_attempts}): {error_msg}")
            
            if attempt < max_attempts - 1:
                print("🔧 Using LLM to repair SQL...")
                current_sql = repair_sql_with_llm(current_sql, error_msg, tokenizer, model, schema, original_question)
                print(f"🔄 Repaired SQL:\n{current_sql}\n")
            else:
                return current_sql, False, attempt + 1
        finally:
            if cursor:
                cursor.close()
    
    return current_sql, False, max_attempts

def run_query(conn, sql: str):
    if sql.strip() == "__SORRY__":
        return "__SORRY__", []
    
    cursor = None
    try:
        conn.ping(reconnect=True, attempts=2, delay=0.5)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        return cols, rows
    except mysql.connector.Error as e:
        raise Exception(f"Query failed: {e}")
    finally:
        if cursor:
            cursor.close()

# --------------------- MAIN ---------------------

def main():
    print("="*60)
    print("INTELLIGENT LEAVE CHATBOT")
    print("="*60)
    
    print("\n🔧 Loading SQL model...")
    tokenizer, model = load_model()
    print("✅ SQL model ready")
    
    intent_detector = IntentDetector()
    
    conn = get_db()
    print("✅ DB connected\n")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            q = input("💬 Your Question: ").strip()
            if q.lower() == "quit":
                break
            
            if not q:
                continue
            
            conn = ensure_connection(conn)
            
            # Preprocess dates
            original_q = q
            q, date_context = preprocess_question(q)
            if date_context:
                print(f"📅 {date_context}")
            if q != original_q:
                print(f"🔄 Preprocessed: {q}")
            
            # Detect intent
            intent, confidence = intent_detector.detect(q)
            print(f"🎯 Intent: {intent} (confidence: {confidence:.2f})")
            
            # Analyze context
            context = analyze_query_context(q)
            print(f"🔍 Query scope: {context['query_scope']}")
            
            # Get schema
            schema = get_schema_for_intent(intent)
            
            # Build prompt
            prompt = build_prompt(q, intent, context)
            
            # Generate SQL
            print("⏳ Generating SQL...")
            raw_sql = generate_sql(tokenizer, model, prompt)
            print(f"\n📝 Generated SQL:\n{raw_sql}\n")
            
            if raw_sql == "__SORRY__":
                print("❌ Cannot generate query")
                continue
            
            # Validate and auto-repair with LLM
            print("🔍 Validating SQL...")
            final_sql, is_valid, attempts = validate_and_repair_sql(
                conn, raw_sql, tokenizer, model, schema, q, max_attempts=2
            )
            
            if is_valid:
                print(f"✅ SQL validated successfully (took {attempts} attempt(s))")
            else:
                print(f"⚠️ SQL validation failed after {attempts} attempts, trying execution anyway...")
            
            # Execute
            print("\n🔄 Executing query...")
            cols, rows = run_query(conn, final_sql)
            
            if isinstance(cols, str) and cols == "__SORRY__":
                print("❌ Execution failed")
                continue
            
            # Display
            if not rows:
                print("📭 No results found")
            else:
                print(f"\n📊 Results ({len(rows)} rows):\n")
                
                col_widths = [max(len(str(c)), max(len(str(row[i])) for row in rows)) for i, c in enumerate(cols)]
                
                header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))
                print(f"  {header}")
                print("  " + "-" * len(header))
                
                for row in rows[:20]:
                    row_str = " | ".join(str(v if v is not None else "NULL").ljust(col_widths[i]) for i, v in enumerate(row))
                    print(f"  {row_str}")
                
                if len(rows) > 20:
                    print(f"\n  ... and {len(rows)-20} more rows")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        conn.close()
    except:
        pass
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
