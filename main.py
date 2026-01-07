from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, render_template_string, request, jsonify, session
from datetime import datetime, timedelta
import secrets
import re
import mysql.connector
from mysql.connector import Error
import pytz

from sentence_transformers import SentenceTransformer, util
import numpy as np


DB_CONFIG = {
    'host': "68.178.155.255",        
    'user': 'Anika12',             
    'password': 'Anika12',    
    'database': 'test_ems'
    
}

def get_db_connection():
    """Create and return MySQL database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

# ===================== DATABASE FUNCTIONS =====================

def get_employee_balance(emp_email):
    """Get leave balance from database for specific employee including LOP"""
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database for leave balance")
            return {'casual': 0, 'sick': 0, 'compOff': 0, 'lop': 0}
        
        cursor = conn.cursor(dictionary=True)
        query = "SELECT cl, sl, co, lop FROM leavebalance WHERE empemail = %s"
        cursor.execute(query, (emp_email,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            # Convert string values to float, handle empty strings
            try:
                cl = float(result['cl']) if result['cl'] and str(result['cl']).strip() != '' else 0.0
            except:
                cl = 0.0
            try:
                sl = float(result['sl']) if result['sl'] and str(result['sl']).strip() != '' else 0.0
            except:
                sl = 0.0
            try:
                co = float(result['co']) if result['co'] and str(result['co']).strip() != '' else 0.0
            except:
                co = 0.0
            try:
                lop = float(result['lop']) if 'lop' in result and result['lop'] and str(result['lop']).strip() != '' else 0.0
            except:
                lop = 0.0
                
            return {
                'casual': cl,
                'sick': sl,
                'compOff': co,
                'lop': lop
            }
        else:
            # If employee not found, create default entry
            print(f"⚠️ Employee {emp_email} not found in leavebalance table")
            return {'casual': 0, 'sick': 0, 'compOff': 0, 'lop': 0}
            
    except Error as e:
        print(f"❌ Database error fetching leave balance: {e}")
        return {'casual': 0, 'sick': 0, 'compOff': 0, 'lop': 0}

def update_employee_balance(emp_email, casual_days, sick_days, comp_off_days, lop_days=0):
    """Update leave balance in database - handle LOP separately, never negative balances"""
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database to update leave balance")
            return False
        
        cursor = conn.cursor()
        
        # Check if employee exists in leavebalance table
        check_query = "SELECT COUNT(*) FROM leavebalance WHERE empemail = %s"
        cursor.execute(check_query, (emp_email,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Insert new record with LOP
            insert_query = """
            INSERT INTO leavebalance (empemail, cl, sl, co, lop, lastupdate) 
            VALUES (%s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(insert_query, (emp_email, str(casual_days), str(sick_days), 
                                        str(comp_off_days), str(lop_days)))
        else:
            # Update existing record with LOP
            update_query = """
            UPDATE leavebalance 
            SET cl = %s, sl = %s, co = %s, lop = %s, lastupdate = NOW()
            WHERE empemail = %s
            """
            cursor.execute(update_query, (str(casual_days), str(sick_days), 
                                        str(comp_off_days), str(lop_days), emp_email))
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Updated leave balance for {emp_email}: CL={casual_days}, SL={sick_days}, CO={comp_off_days}, LOP={lop_days}")
        return True
        
    except Error as e:
        print(f"❌ Database error updating leave balance: {e}")
        return False

def insert_leave_record(leave_data, emp_email, status=0, auto_approval_type=None, approved_days=0, lop_days=0):
    """Insert leave application into leaves table with LOP info"""
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database to insert leave application")
            return None
        
        cursor = conn.cursor()
        
        # Get employee details from leavebalance table
        emp_query = """
        SELECT empname FROM leavebalance WHERE empemail = %s
        """
        cursor.execute(emp_query, (emp_email,))
        emp_result = cursor.fetchone()
        emp_name = emp_result[0] if emp_result else "Unknown"
        
        # Insert leave application
        insert_query = """
        INSERT INTO leaves (
            empname, leavetype, `from`, `to`, reason, empemail, status,
            leavebal, leavetype2, work_location, type_of_auto_approval, approved_days, lop_days
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Determine leave type for database
        leave_type_db = leave_data.get('leaveType', '')
        if leave_type_db == 'casual':
            leave_type_db = 'Casual Leave'
        elif leave_type_db == 'sick':
            leave_type_db = 'Sick Leave'
        elif leave_type_db == 'compOff':
            leave_type_db = 'Comp Off'
        
        # Get current leave balance
        current_balance = get_employee_balance(emp_email)
        
        # Prepare values
        values = (
            emp_name,
            leave_type_db,
            leave_data.get('fromDate'),
            leave_data.get('toDate'),
            leave_data.get('reason', ''),
            emp_email,
            status,  # 0=HR Review, 1=Approved, 2=Rejected, 4=Manager Review
            f"{current_balance['casual']}/{current_balance['sick']}/{current_balance['compOff']}",
            leave_type_db,  # leavetype2 same as leavetype
            'Work Location',  # Default work location
            auto_approval_type,  # 'EMERGENCY' or 'NORMAL' or None
            approved_days,  # Store actual auto-approved days count
            lop_days  # NEW: Store LOP days for this application
        )
        
        cursor.execute(insert_query, values)
        leave_id = cursor.lastrowid
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Inserted leave application ID: {leave_id}, Auto-approval type: {auto_approval_type}, Auto-approved days: {approved_days}, LOP days: {lop_days}")
        return leave_id
        
    except Error as e:
        print(f"❌ Database error inserting leave application: {e}")
        return None

def update_leave_status_db(leave_id, status, remarks="", approved_by="", auto_approval_type=None):
    """Update leave application status in database"""
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database to update leave status")
            return False
        
        cursor = conn.cursor()
        
        if status == 1:  # Approved
            if auto_approval_type:
                # Auto-approved with type
                update_query = """
                UPDATE leaves 
                SET status = %s, status1 = %s, aprtime = NOW(), aprname = %s, type_of_auto_approval = %s
                WHERE ID = %s
                """
                cursor.execute(update_query, (status, remarks, approved_by, auto_approval_type, leave_id))
            else:
                # Regular approval (not auto-approved)
                update_query = """
                UPDATE leaves 
                SET status = %s, status1 = %s, aprtime = NOW(), aprname = %s
                WHERE ID = %s
                """
                cursor.execute(update_query, (status, remarks, approved_by, leave_id))
        elif status == 2:  # Rejected
            update_query = """
            UPDATE leaves 
            SET status = %s, status2 = %s, hrtime = NOW(), hrname = %s
            WHERE ID = %s
            """
            cursor.execute(update_query, (status, remarks, approved_by, leave_id))
        elif status == 0:  # HR Review
            update_query = """
            UPDATE leaves 
            SET status = %s, status1 = %s
            WHERE ID = %s
            """
            cursor.execute(update_query, (status, remarks, leave_id))
        elif status == 4:  # Manager Review
            update_query = """
            UPDATE leaves 
            SET status = %s, status1 = %s
            WHERE ID = %s
            """
            cursor.execute(update_query, (status, remarks, leave_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Updated leave ID {leave_id} to status {status}, Auto-approval type: {auto_approval_type}")
        return True
        
    except Error as e:
        print(f"❌ Database error updating leave status: {e}")
        return False
    
def get_monthly_auto_approved_days(emp_email, month_str):
    """
    Get TOTAL auto-approved DAYS (not count of applications) for a specific month.
    month_str format: 'YYYY-MM'
    Uses the auto_approved_days column to get actual count.
    """
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database for monthly auto-approved count")
            return 0.0  # Return float
        
        cursor = conn.cursor()
        
        # NEW: Sum the auto_approved_days column directly
        query = """
        SELECT COALESCE(SUM(approved_days), 0) FROM leaves 
        WHERE empemail = %s 
        AND YEAR(`from`) = %s AND MONTH(`from`) = %s
        AND type_of_auto_approval IS NOT NULL
        """
        
        year, month = month_str.split('-')
        cursor.execute(query, (emp_email, year, month))
        total_days = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # ✅ FIX: Convert Decimal to float
        total_days = float(total_days) if total_days is not None else 0.0
        
        print(f"📊 Total AUTO-APPROVED days in {month_str}: {total_days}")
        return total_days
        
    except Error as e:
        print(f"❌ Database error getting monthly auto-approved count: {e}")
        return 0

def get_monthly_emergency_requests(emp_email, month_str):
    """
    Get count of emergency REQUESTS (not days) for a specific month.
    RULE: Only 1 emergency request per month allowed.
    """
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database for emergency requests count")
            return 0  # Return int
        
        cursor = conn.cursor()
        
        query = """
        SELECT COUNT(DISTINCT ID) FROM leaves 
        WHERE empemail = %s 
        AND YEAR(`from`) = %s AND MONTH(`from`) = %s
        AND type_of_auto_approval = 'EMERGENCY'
        """
        
        year, month = month_str.split('-')
        cursor.execute(query, (emp_email, year, month))
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # ✅ FIX: Convert Decimal to int
        count = int(count) if count is not None else 0
        
        print(f"📊 Emergency REQUESTS in {month_str}: {count}")
        return count
        
    except Error as e:
        print(f"❌ Database error getting emergency requests: {e}")
        return 0
    
def get_overlapping_leaves(emp_email, from_date_str, to_date_str):
    """
    Check for overlapping leave applications in database
    """
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database for overlapping leaves")
            return []
        
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT * FROM leaves 
        WHERE empemail = %s 
        AND status IN (0, 1, 4)  -- HR Review, Approved, Manager Review
        AND (
            (`from` BETWEEN %s AND %s) 
            OR (`to` BETWEEN %s AND %s)
            OR (%s BETWEEN `from` AND `to`)
            OR (%s BETWEEN `from` AND `to`)
        )
        """
        cursor.execute(query, (emp_email, from_date_str, to_date_str, 
                             from_date_str, to_date_str, from_date_str, 
                             to_date_str))
        
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return results
        
    except Error as e:
        print(f"❌ Database error getting overlapping leaves: {e}")
        return []

def get_holidays_in_range(from_date_str, to_date_str):
    """
    Fetch holidays from database where status is NULL.
    Returns set of holiday dates for fast lookup.
    """
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database for holidays")
            return set()
        
        cursor = conn.cursor()
        
        query = """
            SELECT date 
            FROM holiday 
            WHERE date BETWEEN %s AND %s 
            AND status IS NULL
        """
        
        cursor.execute(query, (from_date_str, to_date_str))
        results = cursor.fetchall()
        
        holiday_dates = set()
        for row in results:
            if row[0]:
                if isinstance(row[0], str):
                    holiday_date = datetime.strptime(row[0], '%Y-%m-%d')
                    holiday_dates.add(holiday_date.date())
                else:
                    holiday_dates.add(row[0])
        
        cursor.close()
        conn.close()
        
        print(f"✅ Fetched {len(holiday_dates)} holidays from database")
        return holiday_dates
        
    except Error as e:
        print(f"❌ Database error: {e}")
        return set()
    except Exception as e:
        print(f"❌ Error fetching holidays: {e}")
        return set()

def get_recent_leave_patterns(emp_email, days_window=60):
    """
    Get recent leave patterns for pattern detection from database
    Returns list of reasons from last 'days_window' days
    """
    try:
        conn = get_db_connection()
        if not conn:
            print("⚠️ Could not connect to database for pattern detection")
            return []
        
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_window)).strftime('%Y-%m-%d')
        
        query = """
        SELECT reason, `from` 
        FROM leaves 
        WHERE empemail = %s 
        AND `from` >= %s
        AND status = 1  -- Only approved leaves
        ORDER BY `from` DESC
        """
        
        cursor.execute(query, (emp_email, cutoff_date))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return results
        
    except Error as e:
        print(f"❌ Database error getting recent leave patterns: {e}")
        return []

# ===================== AI FUNCTIONS =====================

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Loading semantic model for pattern detection...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

CATEGORY_PROTOTYPES = {
    'death': [
        "grandfather passed away",
        "grandmother died",
        "father death",
        "mother funeral",
        "relative died",
        "family member passed away",
        "attending funeral",
        "last rites ceremony",
        "condolence ceremony",
        "dad expired",
        "mom passed away"
    ],
    'marriage': [
        "attending wedding",
        "marriage ceremony",
        "brother marriage",
        "sister wedding",
        "family marriage function",
        "cousin marriage",
        "friend marriage",
        "wedding preparation",
        "marriage reception",
        "own marriage",
        "engagement ceremony",
        "marriage anniversary"
    ],
    'family-illness': [
        "mother is sick",
        "father hospitalized",
        "grandmother in hospital",
        "grandfather surgery",
        "wife is ill",
        "child is sick",
        "family member hospitalized",
        "taking care of sick mother",
        "father medical treatment",
        "relative hospitalized",
        "son has fever",
        "daughter surgery",
        "mother doctor appointment"
    ],
    'travel': [
        "going on trip",
        "family vacation",
        "travel plans",
        "visiting hometown",
        "out of station",
        "going outstation",
        "pilgrimage trip",
        "holiday travel",
        "tour",
        "visiting relatives in another city"
    ],
    'personal': [
        "personal work",
        "personal matter",
        "personal appointment",
        "bank work",
        "government work",
        "documentation work",
        "house work",
        "property matter",
        "legal work",
        "urgent personal work",
        "personal errand"
    ],
    'family-event': [
        "family function",
        "family gathering",
        "family event",
        "house warming ceremony",
        "religious ceremony",
        "birthday celebration",
        "anniversary function",
        "family reunion",
        "naming ceremony",
        "festival celebration"
    ],
    'education': [
        "parent teacher meeting",
        "admission work",
        "child school function",
        "exam supervision",
        "graduation ceremony",
        "attending seminar",
        "training program"
    ],
    'medical': [
        "i have fever",
        "i am sick",
        "i am not feeling well",
        "i have covid",
        "i have headache",
        "i have cold",
        "i have flu",
        "i have stomach pain",
        "medical checkup",
        "doctor appointment",
        "health issue"
    ]
}

print("Precomputing category embeddings...")
CATEGORY_EMBEDDINGS = {}
for category, examples in CATEGORY_PROTOTYPES.items():
    embeddings = semantic_model.encode(examples, convert_to_tensor=True)
    CATEGORY_EMBEDDINGS[category] = embeddings
print("Semantic pattern detection model ready!")

def categorize_reason_for_pattern(reason: str) -> str:
    """
    Use semantic similarity to categorize leave reasons into patterns.
    Returns a category string that captures the semantic meaning of the reason.
    """
    if not reason or len(reason.strip()) < 3:
        return 'other'
    
    reason_cleaned = reason.strip().lower()
    
    reason_embedding = semantic_model.encode(reason_cleaned, convert_to_tensor=True)
    
    best_category = 'other'
    best_similarity = 0.0
    similarity_threshold = 0.45  
    
    for category, prototype_embeddings in CATEGORY_EMBEDDINGS.items():
        similarities = util.cos_sim(reason_embedding, prototype_embeddings)[0]
        max_similarity = float(similarities.max())
        
        if max_similarity > best_similarity:
            best_similarity = max_similarity
            best_category = category
    
    if best_similarity < similarity_threshold:
        return 'other'
        
    return best_category

def detect_reason_pattern_db(emp_email, leave_data, days_window=60, threshold=3):
    """
    Enhanced pattern detection using semantic similarity with database.
    
    Returns (pattern_detected: bool, category: str, count: int).
    Only considers casual reasons (per requirement).
    threshold: number of repeats AFTER which KPI is affected (count > threshold triggers).
    """
    reason = (leave_data.get('reason') or '').strip()
    if not reason:
        return (False, None, 0)

    classification = classify_reason(reason)
    if classification != 'casual':
        return (False, None, 0)

    category = categorize_reason_for_pattern(reason)
    if not category or category == 'other':
        return (False, None, 0)

    count = 1  # Start with current request
    
    # Get recent leave patterns from database
    recent_leaves = get_recent_leave_patterns(emp_email, days_window)
    
    for past_reason_str, past_date in recent_leaves:
        if not past_reason_str:
            continue
            
        past_cat = categorize_reason_for_pattern(past_reason_str)
        
        if past_cat == category:
            count += 1
    
    return (count > threshold, category, count)

def classify_reason(reason: str) -> str:
    """
    Returns 'sick', 'casual', or 'invalid' using Flan-T5 with comprehensive few-shot learning.
    """
    if not reason or len(reason.strip()) < 3:
        return "invalid"
    
    prompt = f"""You are a leave classifier for an employee leave management system.

Classify the following reason into one of three categories:

1. "sick" - ONLY when the EMPLOYEE THEMSELVES is physically ill, injured, or has a medical condition
   Examples: "I have fever", "I am sick", "I injured my leg", "I have COVID", "I am not feeling well"
   
2. "casual" - When the employee is NOT sick themselves (personal work, family matters, travel, events, someone else is sick)
   Examples: "My mother is sick", "Grandmother hospitalized", "Family function", "Personal work", "Father has surgery"
   
3. "invalid" - When the reason is meaningless, too vague or gibberish
   Examples: "xyz", "abc123", "test", "leave", "asdf", "12345"

CRITICAL RULES - READ CAREFULLY:
- Keywords like "mother", "father", "grandmother", "grandfather", "son", "daughter", "wife", "husband", "child", "friend", "relative", "family member" followed by "sick", "ill", "hospitalized", "surgery" = CASUAL LEAVE
- ONLY classify as "sick" if the EMPLOYEE says "I am sick", "I have fever", "I am injured", etc.
- "My [family member] is sick/hospitalized" = CASUAL LEAVE
- "I need to visit [family member] in hospital" = CASUAL LEAVE
- "I need to take care of sick [family member]" = CASUAL LEAVE

Examples for clarity:
- "I am sick" -> sick
- "I have fever" -> sick
- "My grandmother is sick" -> casual
- "My mother is hospitalized" -> casual
- "Need to visit grandmother in hospital" -> casual
- "My father has surgery" -> casual
- "Taking care of sick mother" -> casual
- "Family function" -> casual
- "Personal work" -> casual

Reason: "{reason}"

Think step by step:
1. Does the reason mention "I" or "me" being sick/ill/injured? If yes -> sick
2. Does the reason mention someone else (family member, friend) being sick? If yes -> casual
3. Is the reason too vague or meaningless? If yes -> invalid

Respond with exactly one word: sick, casual, or invalid."""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=10, temperature=0.0)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    if "sick" in result:
        return "sick"
    elif "casual" in result:
        return "casual"
    else:
        return "invalid"

def is_emergency_leave_ai(reason: str, from_date_str: str = None, leave_type: str = None) -> bool:
    """
    Use AI to detect if reason qualifies as emergency with strict criteria.
    Returns False immediately if from_date is provided and not today (RULE 13).
    
    STRICT EMERGENCY CRITERIA:
    - Must be unforeseen/sudden
    - Requires IMMEDIATE action TODAY
    - Cannot be planned or scheduled
    """
    if leave_type == 'sick':
        print(f"❌ Emergency check FAILED: Sick leave cannot be emergency")
        return False
    if not reason or len(reason.strip()) < 5:
        print(f"❌ Emergency check: Reason too short")
        return False
    
    # FIRST CHECK: If from_date is provided, it MUST be TODAY for emergency
    if from_date_str:
        try:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d').date()
            today = datetime.now().date()
            
            print(f"\n{'='*60}")
            print(f"📅 DATE CHECK FOR EMERGENCY:")
            print(f"   From Date: {from_date}")
            print(f"   Today: {today}")
            print(f"   Dates match: {from_date == today}")
            print(f"{'='*60}")
            
            if from_date != today:
                print(f"❌ EMERGENCY CHECK FAILED: From date is not today")
                return False
        except Exception as e:
            print(f"❌ Error parsing date: {e}")
            return False
    
    reason_lower = reason.lower().strip()
    print(f"\n{'='*60}")
    print(f"🔍 REASON ANALYSIS FOR EMERGENCY:")
    print(f"   Reason: {reason}")
    print(f"   Leave Type: {leave_type}")
    print(f"{'='*60}")
    
    # ========== IMMEDIATE REJECTION PATTERNS ==========
    # These patterns are NEVER emergencies (even if date is today)
    
    # 1. Marriage/Wedding related (ALWAYS non-emergency)
    marriage_patterns = [
        'marriage', 'wedding', 'engagement', 'anniversary',
        'marriage function', 'wedding ceremony', 'attending marriage',
        'going to marriage', 'marriage of', 'wedding of',
        'brother marriage', 'sister marriage', 'friend marriage',
        'cousin marriage', 'relative marriage', 'family marriage'
    ]
    
    for pattern in marriage_patterns:
        if pattern in reason_lower:
            print(f"❌ IMMEDIATE REJECTION: Marriage/wedding related → NOT EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return False
    
    # 2. Travel/Going somewhere (ALWAYS non-emergency)
    travel_patterns = [
        'going to', 'going for', 'attending', 'visiting',
        'travel', 'trip', 'tour', 'vacation', 'holiday',
        'outstation', 'out of station', 'going out'
    ]
    
    for pattern in travel_patterns:
        if pattern in reason_lower:
            print(f"❌ IMMEDIATE REJECTION: Travel/attending event → NOT EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return False
    
    # 3. Planned events/functions (ALWAYS non-emergency)
    planned_patterns = [
        'function', 'ceremony', 'celebration', 'festival',
        'party', 'get together', 'reunion', 'gathering',
        'planned', 'planning', 'scheduled', 'pre-planned',
        'in advance', 'advance booking', 'booked'
    ]
    
    for pattern in planned_patterns:
        if pattern in reason_lower:
            print(f"❌ IMMEDIATE REJECTION: Planned event → NOT EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return False
    
    # 4. Personal work/errands (ALWAYS non-emergency)
    personal_patterns = [
        'personal work', 'personal matter', 'personal reason',
        'personal issue', 'personal commitment', 'personal',
        'bank work', 'document work', 'paperwork', 'documentation',
        'appointment', 'meeting', 'work', 'errand', 'shopping'
    ]
    
    for pattern in personal_patterns:
        if pattern in reason_lower:
            print(f"❌ IMMEDIATE REJECTION: Personal work → NOT EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return False
    
    # 5. Routine medical (ALWAYS non-emergency)
    routine_medical_patterns = [
        'routine checkup', 'regular checkup', 'doctor appointment',
        'medical appointment', 'dental appointment', 'eye checkup',
        'follow up', 'follow-up', 'check up', 'check-up'
    ]
    
    for pattern in routine_medical_patterns:
        if pattern in reason_lower:
            print(f"❌ IMMEDIATE REJECTION: Routine medical → NOT EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return False
    
    # ========== CRITICAL EMERGENCY PATTERNS ==========
    # These patterns indicate TRUE emergencies (only if date is today)
    
    # 1. Death/Funeral (ALWAYS emergency if date is today)
    death_patterns = [
        'died', 'death', 'passed away', 'expired', 'funeral',
        'last rites', 'condolence', 'demise', 'no more',
        'grandmother died', 'grandfather died', 'father died', 'mother died',
        'friend died', 'relative died', 'family member died'
    ]
    
    for pattern in death_patterns:
        if pattern in reason_lower:
            print(f"✅ CRITICAL EMERGENCY: Death/funeral → EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return True
    
    # 2. Critical medical emergencies (emergency if date is today)
    medical_emergency_patterns = [
        'heart attack', 'stroke', 'serious accident', 'critical condition',
        'life support', 'icu', 'intensive care', 'emergency surgery',
        'serious injury', 'fractured', 'broken bone', 'bleeding profusely'
    ]
    
    for pattern in medical_emergency_patterns:
        if pattern in reason_lower:
            print(f"✅ CRITICAL EMERGENCY: Medical emergency → EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return True
    
    # 3. Safety/security emergencies (emergency if date is today)
    safety_emergency_patterns = [
        'burglary', 'break in', 'robbery', 'theft', 'police called',
        'fire at home', 'gas leak', 'flooding home', 'electrical fire',
        'evacuate', 'emergency services', 'ambulance', 'rescue'
    ]
    
    for pattern in safety_emergency_patterns:
        if pattern in reason_lower:
            print(f"✅ CRITICAL EMERGENCY: Safety emergency → EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return True
    
    # 4. Transportation emergencies (emergency if date is today)
    transport_emergency_patterns = [
        'breakdown while coming', 'breakdown on the way', 'accident on the way',
        'vehicle accident', 'car crash', 'stranded', 'stuck', 'cannot reach office',
        'no way to come', 'transport issue', 'road blocked', 'traffic accident'
    ]
    
    for pattern in transport_emergency_patterns:
        if pattern in reason_lower:
            print(f"✅ CRITICAL EMERGENCY: Transportation emergency → EMERGENCY")
            print(f"   Pattern: '{pattern}'")
            print(f"{'='*60}\n")
            return True
    
    # ========== AI EMERGENCY DETECTION ==========
    # If none of the above patterns match, use AI to decide
    
    prompt = f"""You are an EMERGENCY LEAVE DETECTOR for workplace attendance.

Analyze if this leave reason qualifies as an EMERGENCY that requires IMMEDIATE leave TODAY.

EMERGENCY means:
1. UNFORESEEABLE - Could NOT have been known or planned in advance
2. URGENT - Requires IMMEDIATE action TODAY, cannot wait
3. CRITICAL - Serious consequences if not addressed immediately
4. ATTENDANCE-PREVENTING - Physically prevents employee from working TODAY

TRUE EMERGENCIES (only if happening TODAY):
✓ Death in immediate family TODAY (father/mother/grandparent died TODAY)
✓ Life-threatening medical emergency TODAY (heart attack, stroke, serious accident)
✓ Critical home emergency TODAY (fire, flood, burglary, gas leak)
✓ Transportation failure WHILE COMING to office TODAY
✓ Natural disaster affecting safety TODAY

NOT EMERGENCIES (these are NEVER emergencies):
✗ ANY planned events (weddings, functions, ceremonies, celebrations)
✗ ANY travel plans (going somewhere, visiting, trips)
✗ ANY personal work or errands
✗ ANY routine medical appointments
✗ ANY family functions or gatherings
✗ ANY reasons starting with "going to", "attending", "visiting"
✗ ANY reasons that could have been planned in advance

Reason: "{reason}"

Is this TRULY an unforeseen, urgent, critical situation requiring immediate leave TODAY?
If YES → EMERGENCY
If NO or if it's about attending/going to/traveling → NOT_EMERGENCY

Final decision: [EMERGENCY or NOT_EMERGENCY]"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        outputs = model.generate(
            **inputs, 
            max_length=20, 
            temperature=0.1,
            do_sample=False,
            num_beams=3
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
        
        print(f"   AI Result: {result}")
        
        emergency_indicators = ["EMERGENCY", "YES", "TRUE", "QUALIFIES", "CRITICAL"]
        non_emergency_indicators = ["NOT_EMERGENCY", "NO", "FALSE", "DOES NOT", "NON-EMERGENCY"]
        
        for indicator in emergency_indicators:
            if indicator in result:
                print(f"   ✅ AI Classified as EMERGENCY")
                print(f"{'='*60}\n")
                return True
        
        for indicator in non_emergency_indicators:
            if indicator in result:
                print(f"   ❌ AI Classified as NOT EMERGENCY")
                print(f"{'='*60}\n")
                return False
        
        print(f"   ⚠️ Unclear AI result, defaulting to NOT EMERGENCY")
        print(f"{'='*60}\n")
        return False
        
    except Exception as e:
        print(f"Error in emergency AI detection: {e}")
        print(f"   Defaulting to NOT EMERGENCY")
        print(f"{'='*60}\n")
        return False


# ===================== DATE VALIDATION FUNCTIONS =====================

def is_valid_date_format(date_string):
    """Check if date is in YYYY-MM-DD format"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def is_valid_casual_comp_off_from_date(date_string, is_emergency=False, leave_type='casual', reason=''):
    """
    Check if FROM date is valid for Casual/Comp Off leave.
    
    Rules:
    - Emergency (casual/compOff): FROM date MUST be TODAY only
    - Normal Casual: FROM date must be tomorrow or future (CANNOT BE TODAY)
    - Normal Comp Off with CASUAL reason: FROM date must be tomorrow or future (CANNOT BE TODAY)
    - Normal Comp Off with SICK reason: FROM date CAN be TODAY (like sick leave)
    """
    try:
        input_date = datetime.strptime(date_string, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        if is_emergency:
            # Emergency MUST be today only (for both casual and compOff)
            return input_date == today
        else:
            # Check if comp off with sick reason
            if leave_type == 'compOff' and reason:
                classification = classify_reason(reason)
                if classification == 'sick':
                    # Comp off with sick reason: can be today or future
                    return input_date >= today
            
            # Normal casual/comp off with casual reason: tomorrow or future (CANNOT BE TODAY)
            return input_date >= tomorrow
            
    except ValueError:
        return False

def validate_compoff_leave_date(date_string, is_emergency=False, reason=''):
    """
    Specific validation for comp off leave.
    
    Rules:
    - Emergency comp off: FROM date MUST be TODAY
    - Comp off with SICK reason: FROM date can be past, today, or future (like sick leave)
    - Comp off with CASUAL reason: FROM date must be TOMORROW or future (CANNOT BE TODAY)
    """
    try:
        input_date = datetime.strptime(date_string, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        print(f"\n{'='*60}")
        print(f"📅 COMP OFF LEAVE DATE VALIDATION:")
        print(f"   Input Date: {input_date.date()}")
        print(f"   Today: {today.date()}")
        print(f"   Tomorrow: {tomorrow.date()}")
        print(f"   Is Emergency: {is_emergency}")
        
        if is_emergency:
            # Emergency MUST be today only
            result = input_date == today
            print(f"   Emergency check: Date must be today → {result}")
        else:
            # Check reason classification
            classification = classify_reason(reason) if reason else 'casual'
            print(f"   Reason Classification: {classification}")
            
            if classification == 'sick':
                # Sick reason: can be past, today, or future (like sick leave)
                result = input_date <= tomorrow  # Past, today, or tomorrow
                print(f"   Comp off with SICK reason: Date can be past/today/tomorrow → {result}")
            else:
                # Casual reason: must be tomorrow or later (CANNOT BE TODAY)
                result = input_date >= tomorrow
                print(f"   Comp off with CASUAL reason: Date must be tomorrow or later (NOT today) → {result}")
        
        print(f"   Final Result: {'VALID' if result else 'INVALID'}")
        print(f"{'='*60}\n")
        
        return result
            
    except ValueError:
        print(f"❌ Invalid date format")
        return False

def validate_casual_leave_date(date_string, is_emergency=False):
    """
    Specific validation for casual leave with one-day-prior rule.
    
    Rules:
    - Emergency casual: FROM date MUST be TODAY
    - Normal casual: FROM date must be TOMORROW or future (CANNOT BE TODAY)
    """
    try:
        input_date = datetime.strptime(date_string, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        print(f"\n{'='*60}")
        print(f"📅 CASUAL LEAVE DATE VALIDATION:")
        print(f"   Input Date: {input_date.date()}")
        print(f"   Today: {today.date()}")
        print(f"   Tomorrow: {tomorrow.date()}")
        print(f"   Is Emergency: {is_emergency}")
        
        if is_emergency:
            # Emergency MUST be today only
            result = input_date == today
            print(f"   Emergency check: Date must be today → {result}")
        else:
            # Normal casual must be tomorrow or later (CANNOT BE TODAY)
            result = input_date >= tomorrow
            print(f"   Normal casual check: Date must be tomorrow or later (NOT today) → {result}")
        
        print(f"   Final Result: {'VALID' if result else 'INVALID'}")
        print(f"{'='*60}\n")
        
        return result
            
    except ValueError:
        print(f"❌ Invalid date format")
        return False
    
def is_valid_sick_leave_from_date(date_string):
    """Check if FROM date is valid for sick leave (past dates, today, or tomorrow only)"""
    try:
        input_date = datetime.strptime(date_string, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        return input_date <= tomorrow
    except ValueError:
        return False

def is_valid_sick_leave_to_date(date_string, from_date_string):
    """Check if TO date is valid for sick leave (can be any date after FROM date)"""
    try:
        to_date = datetime.strptime(date_string, '%Y-%m-%d')
        from_date = datetime.strptime(from_date_string, '%Y-%m-%d')
        
        return to_date >= from_date
    except ValueError:
        return False

def is_valid_casual_comp_off_to_date(date_string, from_date_string):
    """Check if TO date is valid for Casual/Comp Off leave (not before FROM date)"""
    try:
        to_date = datetime.strptime(date_string, '%Y-%m-%d')
        from_date = datetime.strptime(from_date_string, '%Y-%m-%d')
        
        return to_date >= from_date
    except ValueError:
        return False

def calculate_days(from_date, to_date):
    """Calculate working days excluding Sundays AND holidays"""
    try:
        start_date = datetime.strptime(from_date, '%Y-%m-%d')
        end_date = datetime.strptime(to_date, '%Y-%m-%d')
        
        # Fetch holidays from database
        holidays = get_holidays_in_range(from_date, to_date)
        
        working_days = 0
        sundays_count = 0
        holidays_count = 0
        current_date = start_date
        
        while current_date <= end_date:
            is_sunday = current_date.weekday() == 6
            is_holiday = current_date.date() in holidays
            
            if is_sunday:
                sundays_count += 1
            if is_holiday:
                holidays_count += 1
            
            # Count only if not Sunday and not holiday
            if not is_sunday and not is_holiday:
                working_days += 1
            
            current_date += timedelta(days=1)
        
        # Log calculation
        calendar_days = (end_date - start_date).days + 1
        print(f"📅 {calendar_days} calendar days → {working_days} working days")
        print(f"   Excluded: {sundays_count} Sundays, {holidays_count} Holidays")
        
        return working_days
        
    except Exception as e:
        print(f"❌ Error calculating days: {e}")
        return 0

def get_excluded_days_info(from_date, to_date):
    """Get detailed info about excluded days (Sundays and holidays)"""
    try:
        start_date = datetime.strptime(from_date, '%Y-%m-%d')
        end_date = datetime.strptime(to_date, '%Y-%m-%d')
        
        # Get holidays with names
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT date, value as name 
                FROM holiday 
                WHERE date BETWEEN %s AND %s 
                AND status IS NULL
            """
            cursor.execute(query, (from_date, to_date))
            holiday_records = cursor.fetchall()
            cursor.close()
            conn.close()
        else:
            holiday_records = []
        
        # Convert to set and list
        holiday_dates = set()
        holiday_list = []
        for record in holiday_records:
            if record['date']:
                if isinstance(record['date'], str):
                    h_date = datetime.strptime(record['date'], '%Y-%m-%d').date()
                else:
                    h_date = record['date']
                holiday_dates.add(h_date)
                holiday_list.append((h_date, record.get('name', 'Holiday')))
        
        # Count Sundays and working days
        sundays_count = 0
        working_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            is_sunday = current_date.weekday() == 6
            is_holiday = current_date.date() in holiday_dates
            
            if is_sunday:
                sundays_count += 1
            
            if not is_sunday and not is_holiday:
                working_days += 1
            
            current_date += timedelta(days=1)
        
        calendar_days = (end_date - start_date).days + 1
        
        return {
            'sundays': sundays_count,
            'holidays': len(holiday_dates),
            'holiday_list': holiday_list,
            'working_days': working_days,
            'calendar_days': calendar_days
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'sundays': 0,
            'holidays': 0,
            'holiday_list': [],
            'working_days': 0,
            'calendar_days': 0
        }

# ===================== LEAVE PROCESSING FUNCTIONS =====================

EMPLOYEE_EMAIL = "putsalaharshapriya@gmail.com"  # Default email, can be changed

def calculate_lop_amount(emp_email, lop_days=None, monthly_salary=30000):
    """Calculate Loss of Pay amount - optionally get LOP days from database"""
    if lop_days is None:
        # Get current LOP from database
        balance = get_employee_balance(emp_email)
        lop_days = balance.get('lop', 0)
    
    # Convert lop_days to float to avoid Decimal/float type errors
    lop_days = float(lop_days)
    per_day_salary = monthly_salary / 30
    lop_amount = lop_days * per_day_salary
    return round(lop_amount, 2)

def check_auto_approval_eligibility_db(emp_email, leave_data, balance, is_emergency=False):
    """
    Central function to check if leave can be auto-approved.
    IMPLEMENTING FINAL TABLE RULES:
    Only 1 emergency auto-approval per month.
    """
    days = leave_data.get('duration', 0)
    leave_type = leave_data.get('leaveType')
    from_date_str = leave_data.get('fromDate')
    reason = leave_data.get('reason', '')
    
    # Get leave month for quota tracking
    if days <= 0:
        return (False, "No working days in selected date range (all dates are holidays/Sundays)", 0, 0)
    if from_date_str:
        try:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
            leave_month = from_date.strftime('%Y-%m')
        except:
            leave_month = datetime.now().strftime('%Y-%m')
    else:
        leave_month = datetime.now().strftime('%Y-%m')
    
    # Get monthly counters from database
    monthly_approved_days = get_monthly_auto_approved_days(emp_email, leave_month)
    monthly_emergency_requests = get_monthly_emergency_requests(emp_email, leave_month)
    
    print(f"\n{'='*60}")
    print(f"🔍 AUTO-APPROVAL ELIGIBILITY CHECK:")
    print(f"   Leave Type: {leave_type}")
    print(f"   Is Emergency: {is_emergency}")
    print(f"   Days Requested: {days}")
    print(f"   Monthly approved days: {monthly_approved_days}/2")
    print(f"   Emergency requests used: {monthly_emergency_requests}/1")
    print(f"   Balance - Casual: {balance['casual']}, Sick: {balance['sick']}, CompOff: {balance['compOff']}")
    print(f"{'='*60}\n")
    
    # ========== CHECK ELIGIBILITY (BOTH NORMAL AND EMERGENCY) ==========
    is_eligible, violations = check_leave_eligibility()
    print(f"📊 Eligibility Check:")
    print(f"   Is Eligible: {is_eligible}")
    if not is_eligible:
        print(f"   Violations: {violations}")
    
    # ========== EMERGENCY LEAVE LOGIC ==========
    if is_emergency:
        # RULE 12-13: Emergency FROM date must be TODAY
        if from_date_str:
            try:
                from_date = datetime.strptime(from_date_str,'%Y-%m-%d')
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                if from_date != today:
                    return (False, "Emergency leave FROM date must be TODAY", 0, days)
            except:
                return (False, "Invalid FROM date format", 0, days)
        
        # Calculate total available balance based on PRIORITY ORDER
        if leave_type == 'casual':
            # Casual: Casual → Comp Off
            total_balance = balance.get('casual', 0) + balance.get('compOff', 0)
        elif leave_type == 'sick':
            # Sick: Sick → Comp Off → Casual
            total_balance = balance.get('sick', 0) + balance.get('compOff', 0) + balance.get('casual', 0)
        elif leave_type == 'compOff':
            classification = classify_reason(reason) if reason else 'casual'
            if classification == 'sick':
                # Comp off with SICK reason: Comp Off → Sick → Casual
                total_balance = balance.get('compOff', 0) + balance.get('sick', 0) + balance.get('casual', 0)
            else:
                # Comp off with CASUAL reason: Comp Off → Casual
                total_balance = balance.get('compOff', 0) + balance.get('casual', 0)
        else:
            total_balance = 0
        
        print(f"   Total balance available: {total_balance}")
        
        # ========== CHECK EMERGENCY REQUEST LIMIT ==========
        # RULE: Only 1 emergency request per month allowed
        if monthly_emergency_requests >= 1:
            return (False, f"Emergency request limit exhausted for {leave_month} (max 1 emergency request per month)", 0, days)
        
        # ========== CRITICAL FIX: Check if we have ANY balance at all ==========
        # If total_balance is 0 (NO balance), max auto-approval should be 1 day only
        if total_balance == 0:
            print(f"⚠️ NO LEAVE BALANCE AVAILABLE - Emergency max auto-approval: 1 day only")
            
            # Case 1: No auto-approved days used yet (0 used)
            if monthly_approved_days == 0:
                if days == 1:
                    # 0 used, 1 emergency day → 1 auto-approved (LOP), 0 to HR
                    approved = 1
                    hr_review = 0
                    return (True, "Emergency auto-approved (1 day - No balance)", approved, hr_review)
                elif days >= 2:
                    # 0 used, 2+ emergency days → 1 auto-approved (LOP), remaining to HR
                    approved = 1
                    hr_review = days - approved
                    return (True, f"Emergency partial auto-approved (1 day - No balance)", approved, hr_review)
            
            # Case 2: Some auto-approved days already used
            elif monthly_approved_days == 1:
                if days == 1:
                    # 1 used, 1 emergency day → 1 auto-approved (LOP), 0 to HR
                    approved = 1
                    hr_review = 0
                    return (True, "Emergency auto-approved (1 day - No balance)", approved, hr_review)
                elif days >= 2:
                    # 1 used, 2+ emergency days → 1 auto-approved (LOP), remaining to HR
                    approved = 1
                    hr_review = days - approved
                    return (True, f"Emergency partial auto-approved (1 day - No balance)", approved, hr_review)
            
            # Case 3: 2 auto-approved days already used (FULL quota used)
            elif monthly_approved_days >= 2:
                # Emergency override: can approve 1 day as bonus
                if days == 1:
                    # 2 used, 1 emergency day → 1 override auto-approved (LOP), 0 to HR
                    approved = 1
                    hr_review = 0
                    return (True, "Emergency override auto-approved (1 day bonus - No balance)", approved, hr_review)
                elif days >= 2:
                    # 2 used, 2+ emergency days → 1 override auto-approved (LOP), remaining to HR
                    approved = 1
                    hr_review = days - approved
                    return (True, "Emergency override partial auto-approved (1 day bonus - No balance)", approved, hr_review)
        
        # ========== ELIGIBILITY CHECK FOR EMERGENCY ==========
        if not is_eligible:
            # Not eligible - can only auto-approve 1 day maximum
            print(f"⚠️ Employee NOT eligible for full auto-approval. Emergency override: max 1 day")
            
            if days == 1:
                # 1 day emergency when not eligible → 1 auto-approved
                approved = min(1, total_balance if total_balance > 0 else 1)
                hr_review = 0
                return (True, "Emergency auto-approved (1 day - eligibility override)", approved, hr_review)
            elif days >= 2:
                # 2+ days emergency when not eligible → 1 auto-approved, rest to HR
                approved = min(1, total_balance if total_balance > 0 else 1)
                hr_review = days - approved
                return (True, f"Emergency partial auto-approved (1 day - eligibility override)", approved, hr_review)
        
        # ========== ELIGIBLE EMPLOYEE EMERGENCY LOGIC ==========
        # Case 1: No auto-approved days used yet (0 used)
        if monthly_approved_days == 0:
            if days == 1:
                # 0 used, 1 emergency day → 1 auto-approved, 0 to HR
                approved = min(1, total_balance if total_balance > 0 else 1)
                hr_review = 0
                return (True, "Emergency auto-approved (1 day)", approved, hr_review)
            elif days == 2:
                # 0 used, 2 emergency days → 2 auto-approved, 0 to HR
                approved = min(2, total_balance if total_balance > 0 else 2)
                hr_review = 0
                return (True, "Emergency auto-approved (2 days)", approved, hr_review)
            elif days >= 3:
                # 0 used, 3+ emergency days → 2 auto-approved, remaining to HR
                approved = min(2, total_balance if total_balance > 0 else 2)
                hr_review = days - approved
                return (True, f"Emergency partial auto-approved ({approved} days)", approved, hr_review)
        
        # Case 2: 1 auto-approved day already used
        elif monthly_approved_days == 1:
            if days == 1:
                # 1 used, 1 emergency day → 1 auto-approved, 0 to HR
                approved = min(1, total_balance if total_balance > 0 else 1)
                hr_review = 0
                return (True, "Emergency auto-approved (1 day)", approved, hr_review)
            elif days == 2:
                # 1 used, 2 emergency days → 1 auto-approved, 1 to HR
                approved = min(1, total_balance if total_balance > 0 else 1)
                hr_review = days - approved
                return (True, "Emergency partial auto-approved (1 day)", approved, hr_review)
            elif days >= 3:
                # 1 used, 3+ emergency days → 1 auto-approved, remaining to HR
                approved = min(1, total_balance if total_balance > 0 else 1)
                hr_review = days - approved
                return (True, "Emergency partial auto-approved (1 day)", approved, hr_review)
        
        # Case 3: 2 auto-approved days already used (FULL quota used)
        elif monthly_approved_days >= 2:
            # Emergency override: can approve 1 day as bonus
            if days == 1:
                # 2 used, 1 emergency day → 1 override auto-approved, 0 to HR
                approved = 1  # Emergency override always 1 day
                hr_review = 0
                return (True, "Emergency override auto-approved (1 day bonus)", approved, hr_review)
            elif days == 2:
                # 2 used, 2 emergency days → 1 override auto-approved, 1 to HR
                approved = 1
                hr_review = days - approved
                return (True, "Emergency override partial auto-approved (1 day bonus)", approved, hr_review)
            elif days >= 3:
                # 2 used, 3+ emergency days → 1 override auto-approved, remaining to HR
                approved = 1
                hr_review = days - approved
                return (True, "Emergency override partial auto-approved (1 day bonus)", approved, hr_review)
    
    # ========== NORMAL LEAVE LOGIC ==========
    else:
        # RULE 1: Monthly auto-approval limit
        if monthly_approved_days >= 2:
            return (False, f"Monthly auto-approval limit exhausted for {leave_month} (2 days)", 0, days)
        
        # Calculate remaining normal auto-approval days
        remaining_normal_days = 2 - monthly_approved_days
        
        # Calculate available balance based on PRIORITY ORDER
        if leave_type == 'casual':
            # Casual: Casual → Comp Off
            total_available = balance['casual'] + balance['compOff']
        elif leave_type == 'sick':
            # Sick: Sick → Comp Off → Casual
            total_available = balance['sick'] + balance['compOff'] + balance['casual']
        elif leave_type == 'compOff':
            classification = classify_reason(reason) if reason else 'casual'
            if classification == 'sick':
                # Comp off with SICK reason: Comp Off → Sick → Casual
                total_available = balance['compOff'] + balance['sick'] + balance['casual']
            else:
                # Comp off with CASUAL reason: Comp Off → Casual
                total_available = balance['compOff'] + balance['casual']
        else:
            total_available = 0
        
        # RULE 19: Check if we have ANY balance (including from other types)
        if total_available <= 0:
            return (False, "No leave balance available (LOP cannot be auto-approved)", 0, days)
        
        # RULE 4-5: Check eligibility violations
        if not is_eligible:
            return (False, "Eligibility violations: " + "; ".join(violations), 0, days)
        
        # RULE 24-26: Pattern abuse detection
        pattern_detected, pattern_category, pattern_count = detect_reason_pattern_db(
            emp_email, leave_data, days_window=60, threshold=3
        )
        if pattern_detected:
            add_kpi_remark_db(emp_email, f"Semantic pattern abuse detected: {pattern_category} ({pattern_count} times)")
            return (False, f"Pattern abuse detected ({pattern_category} - {pattern_count} occurrences)", 0, days)
        
        # RULE 22-23: Normal leave auto-approval
        if days <= 2:
            approved = min(days, remaining_normal_days, total_available)
            hr_review = days - approved
            return (True, f"Leave auto-approved ({approved} days)", approved, hr_review)
        else:
            # For 3+ days, partial approval up to 2 days
            approved = min(2, remaining_normal_days, total_available)
            hr_review = days - approved
            if approved > 0:
                return (True, f"Partial auto-approval ({approved} of {days} days)", approved, hr_review)
            else:
                return (False, "No auto-approval days available", 0, days)
    return (False, "Unable to process auto-approval", 0, days)

def process_leave_application(emp_email, leave_data):
    """
    Main function to process leave applications with database.
    This function ONLY determines eligibility and returns response.
    It does NOT insert into database until final decision.
    """
    # Get current leave balance from database
    balance = get_employee_balance(emp_email)
    # Convert all values to float to avoid Decimal/float mixing
    balance = {
        'casual': float(balance.get('casual', 0)),
        'sick': float(balance.get('sick', 0)),
        'compOff': float(balance.get('compOff', 0)),
        'lop': float(balance.get('lop', 0))
    }
    
    used_type = ""
    used_parts = []
    leave_type = leave_data.get('leaveType')
    days = float(leave_data.get('duration', 0))  # Convert to float
    reason = leave_data.get('reason', '')
    from_date_str = leave_data.get('fromDate')
    to_date_str = leave_data.get('toDate')
    
    # ========== DETECT IF EMERGENCY ==========
    is_emergency = leave_data.get('isEmergency', False)
    if not is_emergency and from_date_str:
        is_emergency = is_emergency_leave_ai(reason, from_date_str, leave_type)
        leave_data['isEmergency'] = is_emergency
        
    print(f"\n{'='*60}")
    print(f"🔍 PROCESSING LEAVE APPLICATION:")
    print(f"   Leave Type: {leave_type}")
    print(f"   Days: {days}")
    print(f"   Reason: {reason}")
    print(f"   From: {from_date_str}")
    print(f"   To: {to_date_str}")
    print(f"   Is Emergency: {is_emergency}")
    print(f"   Balance - Casual: {balance['casual']}, Sick: {balance['sick']}, CompOff: {balance['compOff']}")
    print(f"{'='*60}\n")
    
    # ========== CHECK ELIGIBILITY ==========
    is_eligible, violations = check_leave_eligibility()
    print(f"📊 Overall Eligibility Status:")
    print(f"   Is Eligible: {is_eligible}")
    if not is_eligible:
        print(f"   Violations: {violations}")
    
    # ========== CHECK AUTO-APPROVAL ELIGIBILITY ==========
    can_auto_approve, reason_msg, approved_days, hr_days = check_auto_approval_eligibility_db(
        emp_email, leave_data, balance, is_emergency
    )
    
    # ========== GET MONTHLY COUNTERS ==========
    if from_date_str:
        try:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
            leave_month = from_date.strftime('%Y-%m')
        except:
            leave_month = datetime.now().strftime('%Y-%m')
    else:
        leave_month = datetime.now().strftime('%Y-%m')
    
    monthly_approved = get_monthly_auto_approved_days(emp_email, leave_month)
    monthly_emergency = get_monthly_emergency_requests(emp_email, leave_month)
    
    # ========== CHECK MONTHLY AUTO-APPROVAL LIMIT ==========
    if monthly_approved >= 2 and not is_emergency:
        return handle_monthly_limit_exhausted_db(emp_email, leave_data, balance, is_emergency)
    
    if not can_auto_approve:
        if "Monthly auto-approval limit exhausted" in reason_msg:
            return handle_monthly_limit_exhausted_db(emp_email, leave_data, balance, is_emergency)
        elif "Insufficient" in reason_msg or "No leave balance available" in reason_msg:
            return suggest_alternative_leaves_db(emp_email, leave_data, balance, is_emergency)
        elif "Pattern abuse" in reason_msg:
            pattern_detected, pattern_category, pattern_count = detect_reason_pattern_db(emp_email, leave_data)
            return {
                'response': f'''⚠️ Your leave cannot be auto-approved due to PATTERN ABUSE.
                
Reason: {reason_msg}

Semantic analysis detected you've used similar reasons {pattern_count} times in the last 60 days.

Your leave application will be sent to Manager/HR for review.''',
                'buttons': [
                    {'text': 'Submit for Approval', 'action': 'submitForApproval'},
                    {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision',
                'leaveData': leave_data
            }
        elif "Eligibility violations" in reason_msg:
            # Check if this is emergency - emergency should bypass eligibility
            if is_emergency:
                # Emergency with eligibility violations - show special message
                return {
                    'response': f'''⚠️ You have eligibility violations, but since this is an EMERGENCY:

{violations}

Only 1 day can be auto-approved for emergency. The remaining {days-1} day(s) require HR approval.''',
                    'buttons': [
                        {'text': 'Proceed with 1 Day Auto-approval', 'action': 'proceedWithEmergencyPartial'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision',
                    'leaveData': leave_data
                }
            else:
                # Normal leave with eligibility violations
                violation_text = "\n".join(violations)
                return {
                    'response': f'''⚠️ Your leave cannot be auto-approved due to eligibility violations:

{violation_text}

Your leave application will be sent to Manager/HR for review.''',
                    'buttons': [
                        {'text': 'Submit for Approval', 'action': 'submitForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision',
                    'leaveData': leave_data
                }
        else:
            if is_emergency:
                recipient = "HR"
            else:
                recipient = "Manager/HR"
            
            return {
                'response': f'''⚠️ Your leave cannot be auto-approved.
                
Reason: {reason_msg}

Your leave application will be sent to {recipient} for review.''',
                'buttons': [
                    {'text': 'Submit for Approval', 'action': 'submitForApproval'},
                    {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision',
                'leaveData': leave_data
            }
    
    # ========== AUTO-APPROVE LOGIC ==========
    if approved_days > 0:
        print(f"📋 AUTO-APPROVAL SUMMARY:")
        print(f"   Approved Days: {approved_days}")
        print(f"   HR Review Days: {hr_days}")
        
        # Convert approved_days and hr_days to float
        approved_days = float(approved_days)
        hr_days = float(hr_days)
        
        # Initialize used_type variable
        used_type = ""
        used_parts = []
        
        # Determine which leave type(s) to use based on PRIORITY ORDER
        new_balance = {
            'casual': float(balance['casual']),
            'sick': float(balance['sick']),
            'compOff': float(balance['compOff']),
            'lop': float(balance['lop'])
        }
        
        remaining_to_approve = float(approved_days)
        lop_to_increment = 0.0
        
        # ========== CHECK REASON FOR COMP OFF ==========
        reason = leave_data.get('reason', '')
        classification = classify_reason(reason) if reason else 'casual'
        
        # ========== NEW DEDUCTION LOGIC WITH LOP COLUMN ==========
        if leave_type == 'casual':
            # CASUAL LEAVE: Casual → Comp Off
            print(f"\n{'='*60}")
            print(f"💰 CASUAL LEAVE DEDUCTION (WITH LOP COLUMN):")
            print(f"   Remaining to approve: {remaining_to_approve}")
            print(f"   Priority: Casual → Comp Off")
            print(f"   Current LOP: {new_balance['lop']}")
            print(f"{'='*60}\n")
            
            # 1. Use Casual Leave first
            casual_available = max(0, new_balance['casual'])
            casual_used = min(casual_available, remaining_to_approve)
            if casual_used > 0:
                used_parts.append(f"{casual_used} Casual Leave")
                new_balance['casual'] -= casual_used
                remaining_to_approve -= casual_used
                print(f"   Used {casual_used} Casual Leave, Remaining: {remaining_to_approve}")
            
            # 2. Use Comp Off second
            if remaining_to_approve > 0:
                compoff_available = max(0, new_balance['compOff'])
                compoff_used = min(compoff_available, remaining_to_approve)
                if compoff_used > 0:
                    used_parts.append(f"{compoff_used} Comp Off")
                    new_balance['compOff'] -= compoff_used
                    remaining_to_approve -= compoff_used
                    print(f"   Used {compoff_used} Comp Off, Remaining: {remaining_to_approve}")
            
            # 3. Any remaining goes to LOP column (NOT negative balances)
            if remaining_to_approve > 0:
                lop_to_increment = remaining_to_approve
                used_parts.append(f"{remaining_to_approve} LOP")
                print(f"   LOP to increment: {lop_to_increment}")
                remaining_to_approve = 0
                
        elif leave_type == 'sick':
            # SICK LEAVE: Sick → Comp Off → Casual
            print(f"\n{'='*60}")
            print(f"💰 SICK LEAVE DEDUCTION (WITH LOP COLUMN):")
            print(f"   Remaining to approve: {remaining_to_approve}")
            print(f"   Priority: Sick → Comp Off → Casual")
            print(f"   Current LOP: {new_balance['lop']}")
            print(f"{'='*60}\n")
            
            # 1. Use Sick Leave first
            sick_available = max(0, new_balance['sick'])
            sick_used = min(sick_available, remaining_to_approve)
            if sick_used > 0:
                used_parts.append(f"{sick_used} Sick Leave")
                new_balance['sick'] -= sick_used
                remaining_to_approve -= sick_used
                print(f"   Used {sick_used} Sick Leave, Remaining: {remaining_to_approve}")
            
            # 2. Use Comp Off second
            if remaining_to_approve > 0:
                compoff_available = max(0, new_balance['compOff'])
                compoff_used = min(compoff_available, remaining_to_approve)
                if compoff_used > 0:
                    used_parts.append(f"{compoff_used} Comp Off")
                    new_balance['compOff'] -= compoff_used
                    remaining_to_approve -= compoff_used
                    print(f"   Used {compoff_used} Comp Off, Remaining: {remaining_to_approve}")
            
            # 3. Use Casual Leave third
            if remaining_to_approve > 0:
                casual_available = max(0, new_balance['casual'])
                casual_used = min(casual_available, remaining_to_approve)
                if casual_used > 0:
                    used_parts.append(f"{casual_used} Casual Leave")
                    new_balance['casual'] -= casual_used
                    remaining_to_approve -= casual_used
                print(f"   Used {casual_used} Casual Leave, Remaining: {remaining_to_approve}")
            
            # 4. Any remaining goes to LOP column (NOT negative balances)
            if remaining_to_approve > 0:
                lop_to_increment = remaining_to_approve
                used_parts.append(f"{remaining_to_approve} LOP")
                print(f"   LOP to increment: {lop_to_increment}")
                remaining_to_approve = 0
                
        elif leave_type == 'compOff':
            # COMP OFF: Check reason classification
            print(f"\n{'='*60}")
            print(f"💰 COMP OFF DEDUCTION (WITH LOP COLUMN):")
            print(f"   Remaining to approve: {remaining_to_approve}")
            print(f"   Reason Classification: {classification}")
            print(f"   Current LOP: {new_balance['lop']}")
            print(f"{'='*60}\n")
            
            if classification == 'sick':
                # Comp Off with SICK reason: Comp Off → Sick → Casual
                print(f"   Priority: Comp Off → Sick → Casual")
                
                # 1. Use Comp Off first
                compoff_available = max(0, new_balance['compOff'])
                compoff_used = min(compoff_available, remaining_to_approve)
                if compoff_used > 0:
                    used_parts.append(f"{compoff_used} Comp Off")
                    new_balance['compOff'] -= compoff_used
                    remaining_to_approve -= compoff_used
                    print(f"   Used {compoff_used} Comp Off, Remaining: {remaining_to_approve}")
                
                # 2. Use Sick Leave second
                if remaining_to_approve > 0:
                    sick_available = max(0, new_balance['sick'])
                    sick_used = min(sick_available, remaining_to_approve)
                    if sick_used > 0:
                        used_parts.append(f"{sick_used} Sick Leave")
                        new_balance['sick'] -= sick_used
                        remaining_to_approve -= sick_used
                        print(f"   Used {sick_used} Sick Leave, Remaining: {remaining_to_approve}")
                
                # 3. Use Casual Leave third
                if remaining_to_approve > 0:
                    casual_available = max(0, new_balance['casual'])
                    casual_used = min(casual_available, remaining_to_approve)
                    if casual_used > 0:
                        used_parts.append(f"{casual_used} Casual Leave")
                        new_balance['casual'] -= casual_used
                        remaining_to_approve -= casual_used
                        print(f"   Used {casual_used} Casual Leave, Remaining: {remaining_to_approve}")
                
                # 4. Any remaining goes to LOP column (NOT negative balances)
                if remaining_to_approve > 0:
                    lop_to_increment = remaining_to_approve
                    used_parts.append(f"{remaining_to_approve} LOP")
                    print(f"   LOP to increment: {lop_to_increment}")
                    remaining_to_approve = 0
            
            else:
                # Comp Off with CASUAL reason: Comp Off → Casual
                print(f"   Priority: Comp Off → Casual")
                
                # 1. Use Comp Off first
                compoff_available = max(0, new_balance['compOff'])
                compoff_used = min(compoff_available, remaining_to_approve)
                if compoff_used > 0:
                    used_parts.append(f"{compoff_used} Comp Off")
                    new_balance['compOff'] -= compoff_used
                    remaining_to_approve -= compoff_used
                    print(f"   Used {compoff_used} Comp Off, Remaining: {remaining_to_approve}")
                
                # 2. Use Casual Leave second
                if remaining_to_approve > 0:
                    casual_available = max(0, new_balance['casual'])
                    casual_used = min(casual_available, remaining_to_approve)
                    if casual_used > 0:
                        used_parts.append(f"{casual_used} Casual Leave")
                        new_balance['casual'] -= casual_used
                        remaining_to_approve -= casual_used
                        print(f"   Used {casual_used} Casual Leave, Remaining: {remaining_to_approve}")
                
                # 3. Any remaining goes to LOP column (NOT negative balances)
                if remaining_to_approve > 0:
                    lop_to_increment = remaining_to_approve
                    used_parts.append(f"{remaining_to_approve} LOP")
                    print(f"   LOP to increment: {lop_to_increment}")
                    remaining_to_approve = 0

        # Ensure no negative balances (safety check)
        new_balance['casual'] = max(0, new_balance['casual'])
        new_balance['sick'] = max(0, new_balance['sick'])
        new_balance['compOff'] = max(0, new_balance['compOff'])

        # Update LOP in balance
        new_balance['lop'] += lop_to_increment
        
        # Build used_type from used_parts
        if used_parts:
            used_type = " + ".join(used_parts)
        else:
            used_type = f"{approved_days} day(s)"
        
        # ========== CRITICAL FIX: AUTO-APPROVE IMMEDIATELY ==========
        # Auto-approve the eligible days automatically (no user confirmation needed)
        
        # Update leave balance in database for auto-approved days WITH LOP
        update_employee_balance(emp_email, new_balance['casual'], 
                              new_balance['sick'], new_balance['compOff'],
                              new_balance['lop'])
        
        # Insert leave application with approved status for auto-approved days
        status = 1  # Approved
        auto_approval_type = 'EMERGENCY' if is_emergency else 'NORMAL'
        
        # Check if we need to ask user about remaining days
        if hr_days > 0:
            # CRITICAL FIX: Insert Record 1 - Auto-approved portion
            leave_id = insert_leave_record(leave_data, emp_email, status, auto_approval_type, 
                                         approved_days=approved_days, lop_days=lop_to_increment)
            
            if leave_id:
                remarks = f"Auto-approved {approved_days} day(s) using {used_type}"
                update_leave_status_db(leave_id, status, remarks, "SYSTEM", auto_approval_type)
                
                # Store leave data for HR/Manager review (Record 2)
                # IMPORTANT: Store the ORIGINAL leave_data for remaining days
                session['remaining_leave_data'] = leave_data.copy()  # Store copy of SAME data
                session['remaining_days'] = float(hr_days)  # Convert to float
                session['already_approved_days'] = float(approved_days)  # Convert to float
                session['is_emergency_leave'] = is_emergency
                session.modified = True  # ✅ CRITICAL: Mark session as modified
                
                # Calculate what balance will be used for remaining days if HR approves
                remaining_balance = new_balance.copy()
                hr_balance_usage = []
                hr_lop_days = 0
                remaining_to_use = float(hr_days)

                leave_type = leave_data.get('leaveType')

                if leave_type == 'casual':
                    # Use casual first, then comp-off
                    casual_avail = max(0, float(remaining_balance['casual']))  # ✅ FIX
                    casual_used = min(casual_avail, remaining_to_use)
                    if casual_used > 0:
                        hr_balance_usage.append(f"{casual_used} Casual Leave")
                        remaining_to_use -= casual_used
                    
                    if remaining_to_use > 0:
                        compoff_avail = max(0, float(remaining_balance['compOff']))  # ✅ FIX
                        compoff_used = min(compoff_avail, remaining_to_use)
                        if compoff_used > 0:
                            hr_balance_usage.append(f"{compoff_used} Comp Off")
                            remaining_to_use -= compoff_used
                    
                    if remaining_to_use > 0:
                        hr_lop_days = remaining_to_use
                        
                elif leave_type == 'sick':
                    # Use sick first, then comp-off, then casual
                    sick_avail = max(0, float(remaining_balance['sick']))  # ✅ FIX
                    sick_used = min(sick_avail, remaining_to_use)
                    if sick_used > 0:
                        hr_balance_usage.append(f"{sick_used} Sick Leave")
                        remaining_to_use -= sick_used
                    
                    if remaining_to_use > 0:
                        compoff_avail = max(0, float(remaining_balance['compOff']))  # ✅ FIX
                        compoff_used = min(compoff_avail, remaining_to_use)
                        if compoff_used > 0:
                            hr_balance_usage.append(f"{compoff_used} Comp Off")
                            remaining_to_use -= compoff_used
                    
                    if remaining_to_use > 0:
                        casual_avail = max(0, float(remaining_balance['casual']))  # ✅ FIX
                        casual_used = min(casual_avail, remaining_to_use)
                        if casual_used > 0:
                            hr_balance_usage.append(f"{casual_used} Casual Leave")
                            remaining_to_use -= casual_used
                    
                    if remaining_to_use > 0:
                        hr_lop_days = remaining_to_use
                        
                elif leave_type == 'compOff':
                    # Check reason classification
                    reason = leave_data.get('reason', '')
                    classification = classify_reason(reason) if reason else 'casual'
                    
                    if classification == 'sick':
                        # Comp off with sick reason: compOff → sick → casual
                        compoff_avail = max(0, float(remaining_balance['compOff']))  # ✅ FIX
                        compoff_used = min(compoff_avail, remaining_to_use)
                        if compoff_used > 0:
                            hr_balance_usage.append(f"{compoff_used} Comp Off")
                            remaining_to_use -= compoff_used
                        
                        if remaining_to_use > 0:
                            sick_avail = max(0, float(remaining_balance['sick']))  # ✅ FIX
                            sick_used = min(sick_avail, remaining_to_use)
                            if sick_used > 0:
                                hr_balance_usage.append(f"{sick_used} Sick Leave")
                                remaining_to_use -= sick_used
                        
                        if remaining_to_use > 0:
                            casual_avail = max(0, float(remaining_balance['casual']))  # ✅ FIX
                            casual_used = min(casual_avail, remaining_to_use)
                            if casual_used > 0:
                                hr_balance_usage.append(f"{casual_used} Casual Leave")
                                remaining_to_use -= casual_used
                    else:
                        # Comp off with casual reason: compOff → casual
                        compoff_avail = max(0, float(remaining_balance['compOff']))  # ✅ FIX
                        compoff_used = min(compoff_avail, remaining_to_use)
                        if compoff_used > 0:
                            hr_balance_usage.append(f"{compoff_used} Comp Off")
                            remaining_to_use -= compoff_used
                        
                        if remaining_to_use > 0:
                            casual_avail = max(0, float(remaining_balance['casual']))  # ✅ FIX
                            casual_used = min(casual_avail, remaining_to_use)
                            if casual_used > 0:
                                hr_balance_usage.append(f"{casual_used} Casual Leave")
                                remaining_to_use -= casual_used
                    
                    if remaining_to_use > 0:
                        hr_lop_days = remaining_to_use
                
                # Build HR balance usage text
                if hr_balance_usage:
                    hr_usage_text = " + ".join(hr_balance_usage)
                    if hr_lop_days > 0:
                        lop_amount = calculate_lop_amount(emp_email, hr_lop_days)
                        hr_usage_text += f" + {hr_lop_days} LOP (₹{lop_amount})"
                else:
                    if hr_lop_days > 0:
                        lop_amount = calculate_lop_amount(emp_email, hr_lop_days)
                        hr_usage_text = f"{hr_lop_days} LOP (₹{lop_amount})"
                    else:
                        hr_usage_text = "available balance"
                
                # Build response message
                recipient = "HR" if is_emergency else "Manager/HR"
                lop_amount = calculate_lop_amount(emp_email, lop_to_increment)
                
                response_msg = f'''✅ {approved_days} day(s) have been auto-approved using {used_type}!

The remaining {hr_days} day(s) require {recipient} approval.

Your updated leave balance (after auto-approval):
📅 Casual Leave: {new_balance['casual']} days
🏥 Sick Leave: {new_balance['sick']} days
⏰ Comp Off: {new_balance['compOff']} days
💰 LOP Days: {new_balance['lop']} days (₹{lop_amount})

📊 Monthly auto-approved days used for {leave_month}: {float(monthly_approved) + approved_days}/2'''
                
                if is_emergency:
                    response_msg += f"\n\n🚨 Emergency requests used for {leave_month}: {int(monthly_emergency) + 1}/1"
                
                # Add HR portion info
                response_msg += f"\n\n📋 If {recipient} approves the remaining {hr_days} day(s), it will use: {hr_usage_text}"
                response_msg += f"\n\nDo you want to submit the remaining {hr_days} day(s) for {recipient} approval?"
                
                return {
                    'response': response_msg,
                    'buttons': [
                        {'text': f'Submit for {recipient} Approval', 'action': 'submitRemainingForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision',
                    'leaveData': leave_data
                }
            else:
                return {
                    'response': '❌ Error processing auto-approval. Please try again.',
                    'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
                    'state': 'initial',
                    'leaveData': {}
                }
        else:
            # All days auto-approved - show success message
            leave_id = insert_leave_record(leave_data, emp_email, 1, 
                                         'EMERGENCY' if is_emergency else 'NORMAL', 
                                         approved_days=approved_days, lop_days=lop_to_increment)
            
            if leave_id:
                remarks = f"Auto-approved {approved_days} day(s) using {used_type}"
                update_leave_status_db(leave_id, 1, remarks, "SYSTEM", 
                                     'EMERGENCY' if is_emergency else 'NORMAL')
                
                lop_amount = calculate_lop_amount(emp_email, lop_to_increment)
                response_msg = f'''✅ {approved_days} day(s) have been auto-approved using {used_type}!

Your updated leave balance:
📅 Casual Leave: {new_balance['casual']} days
🏥 Sick Leave: {new_balance['sick']} days
⏰ Comp Off: {new_balance['compOff']} days
💰 LOP Days: {new_balance['lop']} days (₹{lop_amount})

📊 Monthly auto-approved days used for {leave_month}: {float(monthly_approved) + approved_days}/2'''
                
                if is_emergency:
                    response_msg += f"\n\n🚨 Emergency requests used for {leave_month}: {int(monthly_emergency) + 1}/1"
                
                return {
                    'response': response_msg,
                    'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
                    'state': 'initial',
                    'leaveData': {}
                }
            else:
                return {
                    'response': '❌ Error processing auto-approval. Please try again.',
                    'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
                    'state': 'initial',
                    'leaveData': {}
                }
    
    # If no auto-approval, suggest alternatives
    return suggest_alternative_leaves_db(emp_email, leave_data, balance, is_emergency)

def add_kpi_remark_db(emp_email, remark):
    """Store KPI remarks in database or session"""
    # For now, store in session. You can modify to store in database if needed.
    if 'kpi_remarks' not in session:
        session['kpi_remarks'] = []
    session['kpi_remarks'].append({
        'remark': remark,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    session.modified = True

def check_leave_eligibility():
    """
    Check if employee is eligible for automatic leave approval.
    Returns: (is_eligible: bool, violations: list)
    """
    violations = []
    
    # Rule 1: Last month attendance > 90%
    last_month_attendance = session.get('last_month_attendance', 95.0)
    if last_month_attendance <= 90:
        violations.append(f"❌ Last month attendance ({last_month_attendance}%) is not above 90%")
    
    # Rule 2: Last month work performance > 90%
    last_month_performance = session.get('last_month_performance', 95.0)
    if last_month_performance <= 90:
        violations.append(f"❌ Last month work performance ({last_month_performance}%) is not above 90%")
    
    # Rule 3: Late coming minutes exceeded (120 minutes limit)
    late_minutes_used = session.get('late_minutes_used', 0)
    if late_minutes_used >= 120:
        violations.append(f"❌ Late coming limit exceeded ({late_minutes_used} minutes used, limit: 120 minutes)")
    
    # Rule 4: Last 3 months average attendance > 85%
    last_3_months_attendance = session.get('last_3_months_attendance', 75.0)  # Changed to 75% to match your requirement
    if last_3_months_attendance <= 85:
        violations.append(f"❌ Last 3 months average attendance ({last_3_months_attendance}%) is not above 85%")
    
    # Rule 5: Pattern recognition - this is handled separately via semantic detection
    
    is_eligible = len(violations) == 0
    return is_eligible, violations

def is_employee_sick(reason):
    """
    Decide if the employee is sick using the AI model.
    Returns True if classified as 'sick', else False.
    """
    if not reason:
        return False
    label = classify_reason(reason)
    return label == "sick"

def handle_monthly_limit_exhausted_db(emp_email, leave_data, balance, is_emergency=False):
    """Handle case when monthly auto-approval limit is exhausted"""
    leave_type = leave_data.get('leaveType')
    days = leave_data.get('duration', 0)
    reason = leave_data.get('reason', '')
    
    if leave_data.get('fromDate'):
        try:
            from_date = datetime.strptime(leave_data.get('fromDate'), '%Y-%m-%d')
            leave_month = from_date.strftime('%Y-%m')
        except:
            leave_month = datetime.now().strftime('%Y-%m')
    else:
        leave_month = datetime.now().strftime('%Y-%m')
    
    emergency_requests = get_monthly_emergency_requests(emp_email, leave_month)
    
    if is_emergency:
        # Check if emergency quota available
        if emergency_requests >= 1:
            return {
                'response': f'''❌ Emergency quota exhausted for {leave_month} (max 1 emergency request per month).

Your application cannot be processed.''',
                'buttons': [
                    {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision',
                'leaveData': leave_data
            }
        
        # ⚠️ CRITICAL FIX: Emergency should override monthly limit with 1 day bonus
        if days == 1:
            # Auto-approve 1 day emergency as bonus
            new_balance = {
                'casual': float(balance['casual']),
                'sick': float(balance['sick']),
                'compOff': float(balance['compOff']),
                'lop': float(balance['lop'])
            }
            
            # Calculate what to use based on priority
            lop_to_increment = 0.0
            used_parts = []
            
            if leave_type == 'casual':
                # Casual: Casual → Comp Off
                if new_balance['casual'] >= 1:
                    new_balance['casual'] -= 1
                    used_parts.append(f"1 Casual Leave")
                elif new_balance['compOff'] >= 1:
                    new_balance['compOff'] -= 1
                    used_parts.append(f"1 Comp Off")
                else:
                    lop_to_increment = 1
                    used_parts.append(f"1 LOP")
            
            elif leave_type == 'sick':
                # Sick: Sick → Comp Off → Casual
                if new_balance['sick'] >= 1:
                    new_balance['sick'] -= 1
                    used_parts.append(f"1 Sick Leave")
                elif new_balance['compOff'] >= 1:
                    new_balance['compOff'] -= 1
                    used_parts.append(f"1 Comp Off")
                elif new_balance['casual'] >= 1:
                    new_balance['casual'] -= 1
                    used_parts.append(f"1 Casual Leave")
                else:
                    lop_to_increment = 1
                    used_parts.append(f"1 LOP")
            
            elif leave_type == 'compOff':
                classification = classify_reason(reason) if reason else 'casual'
                if classification == 'sick':
                    # Comp off with SICK reason: Comp Off → Sick → Casual
                    if new_balance['compOff'] >= 1:
                        new_balance['compOff'] -= 1
                        used_parts.append(f"1 Comp Off")
                    elif new_balance['sick'] >= 1:
                        new_balance['sick'] -= 1
                        used_parts.append(f"1 Sick Leave")
                    elif new_balance['casual'] >= 1:
                        new_balance['casual'] -= 1
                        used_parts.append(f"1 Casual Leave")
                    else:
                        lop_to_increment = 1
                        used_parts.append(f"1 LOP")
                else:
                    # Comp off with CASUAL reason: Comp Off → Casual
                    if new_balance['compOff'] >= 1:
                        new_balance['compOff'] -= 1
                        used_parts.append(f"1 Comp Off")
                    elif new_balance['casual'] >= 1:
                        new_balance['casual'] -= 1
                        used_parts.append(f"1 Casual Leave")
                    else:
                        lop_to_increment = 1
                        used_parts.append(f"1 LOP")
            
            # Ensure no negative balances
            new_balance['casual'] = max(0, new_balance['casual'])
            new_balance['sick'] = max(0, new_balance['sick'])
            new_balance['compOff'] = max(0, new_balance['compOff'])
            new_balance['lop'] += lop_to_increment
            
            # Update database
            update_employee_balance(emp_email, new_balance['casual'], 
                                  new_balance['sick'], new_balance['compOff'],
                                  new_balance['lop'])
            
            # Insert leave as auto-approved emergency
            leave_id = insert_leave_record(leave_data, emp_email, 1, 'EMERGENCY', 
                                         approved_days=1.0, lop_days=lop_to_increment)
            
            if leave_id:
                update_leave_status_db(leave_id, 1, "Emergency override (1 day bonus) - Auto-approved", "SYSTEM", 'EMERGENCY')
                
                used_type = " + ".join(used_parts) if used_parts else "1 day"
                lop_amount = calculate_lop_amount(emp_email, lop_to_increment)
                
                return {
                    'response': f'''✅ 1 day emergency leave auto-approved as bonus override!

Since you've used all monthly auto-approval quota (2/2 days),
but haven't used emergency quota, 1 day has been auto-approved as emergency bonus.

Approved using: {used_type}

Your updated leave balance:
📅 Casual Leave: {new_balance['casual']} days
🏥 Sick Leave: {new_balance['sick']} days
⏰ Comp Off: {new_balance['compOff']} days
💰 LOP Days: {new_balance['lop']} days (₹{lop_amount})

🚨 Emergency requests used for {leave_month}: {emergency_requests + 1}/1''',
                    'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
                    'state': 'initial',
                    'leaveData': {}
                }
        
        recipient = "HR"
        status = 0  # HR Review
    else:
        recipient = "Manager/HR"
        status = 4  # Manager Review
    
    # Check what leaves would be used if approved based on PRIORITY ORDER
    parts = []
    remaining = days
    
    if leave_type == 'casual':
        # Casual: Casual → Comp Off
        casual_used = min(balance['casual'], remaining)
        if casual_used > 0:
            parts.append(f"{casual_used} Casual Leave")
            remaining -= casual_used
        
        if remaining > 0:
            compoff_used = min(balance['compOff'], remaining)
            if compoff_used > 0:
                parts.append(f"{compoff_used} Comp Off")
                remaining -= compoff_used
    
    elif leave_type == 'sick':
        # Sick: Sick → Comp Off → Casual
        sick_used = min(balance['sick'], remaining)
        if sick_used > 0:
            parts.append(f"{sick_used} Sick Leave")
            remaining -= sick_used
        
        if remaining > 0:
            compoff_used = min(balance['compOff'], remaining)
            if compoff_used > 0:
                parts.append(f"{compoff_used} Comp Off")
                remaining -= compoff_used
        
        if remaining > 0:
            casual_used = min(balance['casual'], remaining)
            if casual_used > 0:
                parts.append(f"{casual_used} Casual Leave")
                remaining -= casual_used
    
    elif leave_type == 'compOff':
        # Comp Off: Check reason
        classification = classify_reason(reason) if reason else 'casual'
        
        if classification == 'sick':
            # Comp Off with SICK reason: Comp Off → Sick → Casual
            compoff_used = min(balance['compOff'], remaining)
            if compoff_used > 0:
                parts.append(f"{compoff_used} Comp Off")
                remaining -= compoff_used
            
            if remaining > 0:
                sick_used = min(balance['sick'], remaining)
                if sick_used > 0:
                    parts.append(f"{sick_used} Sick Leave")
                    remaining -= sick_used
            
            if remaining > 0:
                casual_used = min(balance['casual'], remaining)
                if casual_used > 0:
                    parts.append(f"{casual_used} Casual Leave")
                    remaining -= casual_used
        else:
            # Comp Off with CASUAL reason: Comp Off → Casual
            compoff_used = min(balance['compOff'], remaining)
            if compoff_used > 0:
                parts.append(f"{compoff_used} Comp Off")
                remaining -= compoff_used
            
            if remaining > 0:
                casual_used = min(balance['casual'], remaining)
                if casual_used > 0:
                    parts.append(f"{casual_used} Casual Leave")
                    remaining -= casual_used
    
    # Add LOP if any remaining days
    if remaining > 0:
        lop_amount = calculate_lop_amount(emp_email, remaining)
        parts.append(f"{remaining} LOP (₹{lop_amount})")
    
    leave_source = " + ".join(parts) if parts else f"{days} LOP (₹{calculate_lop_amount(emp_email, days)})"
    
    # Store leave data for final submission
    session['pending_approval'] = True
    session['needs_review'] = True
    session['review_type'] = 'EMERGENCY' if is_emergency else 'NORMAL'
    
    return {
        'response': f'''📤 Your leave cannot be auto-approved due to monthly limit exhaustion (2/2 days used).

Your application needs to be sent to {recipient} for approval.

If approved, it will use: {leave_source}

Your current balance (unchanged until approval):
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{calculate_lop_amount(emp_email, balance['lop'])})

Do you want to submit for {recipient} approval?''',
        'buttons': [
            {'text': f'Submit for {recipient} Approval', 'action': 'submitForApproval'},
            {'text': 'Cancel Leave', 'action': 'cancelLeave'}
        ],
        'state': 'waitingDecision',
        'leaveData': leave_data
    }

def suggest_alternative_leaves_db(emp_email, leave_data, balance, is_emergency=False):
    """
    Suggest alternative leave options when primary type has insufficient balance.
    """
    leave_type = leave_data.get('leaveType')
    days = float(leave_data.get('duration', 0))  # Convert to float
    reason = leave_data.get('reason', '')
    
    # Check monthly auto-approval limit first
    from_date_str = leave_data.get('fromDate')
    if from_date_str:
        try:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
            leave_month = from_date.strftime('%Y-%m')
        except:
            leave_month = datetime.now().strftime('%Y-%m')
    else:
        leave_month = datetime.now().strftime('%Y-%m')
    
    monthly_approved = get_monthly_auto_approved_days(emp_email, leave_month)
    monthly_emergency = get_monthly_emergency_requests(emp_email, leave_month)
    
    # Check emergency quota
    if is_emergency and monthly_emergency >= 1:
        return {
            'response': f'''❌ Emergency quota exhausted for {leave_month} (max 1 emergency request per month).

Your application cannot be processed.''',
            'buttons': [
                {'text': 'Cancel Leave', 'action': 'cancelLeave'}
            ],
            'state': 'waitingDecision',
            'leaveData': leave_data
        }
    
    if monthly_approved >= 2 and not is_emergency:
        return handle_monthly_limit_exhausted_db(emp_email, leave_data, balance, is_emergency)
    
    # Calculate total available balance based on PRIORITY ORDER
    if leave_type == 'casual':
        # Casual: Casual → Comp Off
        total_available = balance['casual'] + balance['compOff']
        available_types = [
            ('Casual Leave', balance['casual']),
            ('Comp Off', balance['compOff'])
        ]
    elif leave_type == 'sick':
        # Sick: Sick → Comp Off → Casual
        total_available = balance['sick'] + balance['compOff'] + balance['casual']
        available_types = [
            ('Sick Leave', balance['sick']),
            ('Comp Off', balance['compOff']),
            ('Casual Leave', balance['casual'])
        ]
    elif leave_type == 'compOff':
        classification = classify_reason(reason) if reason else 'casual'
        
        if classification == 'sick':
            # Comp off with SICK reason: Comp Off → Sick → Casual
            total_available = balance['compOff'] + balance['sick'] + balance['casual']
            available_types = [
                ('Comp Off', balance['compOff']),
                ('Sick Leave', balance['sick']),
                ('Casual Leave', balance['casual'])
            ]
        else:
            # Comp off with CASUAL reason: Comp Off → Casual
            total_available = balance['compOff'] + balance['casual']
            available_types = [
                ('Comp Off', balance['compOff']),
                ('Casual Leave', balance['casual'])
            ]
    else:
        total_available = 0
        available_types = []
    
    # Handle NO balance at all
    if total_available <= 0:
        if is_emergency and monthly_approved == 0:
            # No balance, no monthly quota used: 1 day emergency auto-approved
            # Update balance with LOP
            new_balance = {
                'casual': float(balance['casual']),
                'sick': float(balance['sick']),
                'compOff': float(balance['compOff']),
                'lop': float(balance['lop'])
            }
            
            lop_to_increment = 1.0
            new_balance['lop'] += lop_to_increment
            
            # Update database
            update_employee_balance(emp_email, new_balance['casual'], 
                                  new_balance['sick'], new_balance['compOff'],
                                  new_balance['lop'])
            
            # Insert leave as auto-approved emergency
            status = 1  # Approved
            leave_id = insert_leave_record(leave_data, emp_email, status, 'EMERGENCY', 
                                         approved_days=1.0, lop_days=lop_to_increment)
            
            if leave_id:
                update_leave_status_db(leave_id, status, "Emergency override with LOP - Auto-approved", "SYSTEM", 'EMERGENCY')
                
                lop_amount = calculate_lop_amount(emp_email, lop_to_increment)
                return {
                    'response': f'''✅ 1 day emergency leave auto-approved!

Since you had no leave balance but haven't used any monthly quota,
1 day has been auto-approved with Loss of Pay (₹{lop_amount}).

Your updated leave balance:
📅 Casual Leave: {new_balance['casual']} days
🏥 Sick Leave: {new_balance['sick']} days
⏰ Comp Off: {new_balance['compOff']} days
💰 LOP Days: {new_balance['lop']} days (₹{lop_amount})

🚨 Emergency request quota used: 1/1''',
                    'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
                    'state': 'initial',
                    'leaveData': {}
                }
            else:
                return {
                    'response': '❌ Error processing emergency approval.',
                    'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
                    'state': 'initial',
                    'leaveData': {}
                }
        else:
            # No emergency or quota used: send for review
            recipient = "HR" if is_emergency else "Manager/HR"
            status = 0 if is_emergency else 4
            
            lop_amount = calculate_lop_amount(emp_email, days)
            
            session['pending_approval'] = True
            session['needs_review'] = True
            session['review_type'] = 'EMERGENCY' if is_emergency else 'NORMAL'
            
            return {
                'response': f'''⚠️ You don't have sufficient leave balance for any leave type.

Current balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{calculate_lop_amount(emp_email, balance['lop'])})

This will affect your KPI and ₹{lop_amount} LOP will be deducted from your salary for {days} day(s).
This request requires {recipient} approval (LOP is never auto-approved).

Do you want to proceed to {recipient} review?''',
                'buttons': [
                    {'text': f'Submit for {recipient} Approval', 'action': 'submitForApproval'},
                    {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision',
                'leaveData': leave_data
            }
    
    # There is some balance available
    available_in_current = total_available
    
    if available_in_current > 0:
        shortfall = max(0, days - available_in_current)
        
        if shortfall > 0:
            # Partial LOP required
            lop_amount = calculate_lop_amount(emp_email, shortfall)
            
            # Check eligibility for auto-approval
            if is_emergency:
                # Emergency should bypass eligibility checks
                return process_leave_application(emp_email, leave_data)
            else:
                # Normal leave - check eligibility
                is_eligible, violations = check_leave_eligibility()
                if not is_eligible:
                    violation_text = "\n".join(violations)
                    return {
                        'response': f'''⚠️ You have some leave balance, but partial LOP is required:

Balance available:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{calculate_lop_amount(emp_email, balance['lop'])})

Shortfall: {shortfall} day(s) LOP (₹{lop_amount})

However, you are NOT eligible for auto-approval due to:

{violation_text}

This request will be sent to Manager/HR for review.''',
                        'buttons': [
                            {'text': 'Submit for Approval', 'action': 'submitForApproval'},
                            {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                        ],
                        'state': 'waitingDecision',
                        'leaveData': leave_data
                    }
                else:
                    # Eligible - use main processing function
                    return process_leave_application(emp_email, leave_data)
        else:
            # Enough balance available
            if is_emergency:
                # Emergency - use main processing function
                return process_leave_application(emp_email, leave_data)
            else:
                # Normal leave - check eligibility
                is_eligible, violations = check_leave_eligibility()
                if not is_eligible:
                    violation_text = "\n".join(violations)
                    return {
                        'response': f'''⚠️ You have sufficient leave balance, but you are NOT eligible for auto-approval:

{violation_text}

This request will be sent to Manager/HR for review.''',
                        'buttons': [
                            {'text': 'Submit for Approval', 'action': 'submitForApproval'},
                            {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                        ],
                        'state': 'waitingDecision',
                        'leaveData': leave_data
                    }
                else:
                    # Eligible - use main processing function
                    return process_leave_application(emp_email, leave_data)
    
    # Check total available balance
    total_available = available_in_current
    if total_available > 0:
        return check_partial_lop_options_db(emp_email, leave_data, balance, is_emergency)
    else:
        # NO balance at all - handled above
        pass

def check_partial_lop_options_db(emp_email, leave_data, balance, is_emergency=False):
    """Check if partial LOP is possible with available balances"""
    days = leave_data.get('duration', 0)
    leave_type = leave_data.get('leaveType')
    reason = leave_data.get('reason', '')
    
    # Build available types list based on PRIORITY ORDER
    available_types = []
    
    if leave_type == 'casual':
        # Casual: Casual → Comp Off
        if balance['casual'] > 0:
            available_types.append(('Casual Leave', balance['casual']))
        if balance['compOff'] > 0:
            available_types.append(('Comp Off', balance['compOff']))
        total_available = balance['casual'] + balance['compOff']
        
    elif leave_type == 'sick':
        # Sick: Sick → Comp Off → Casual
        if balance['sick'] > 0:
            available_types.append(('Sick Leave', balance['sick']))
        if balance['compOff'] > 0:
            available_types.append(('Comp Off', balance['compOff']))
        if balance['casual'] > 0:
            available_types.append(('Casual Leave', balance['casual']))
        total_available = balance['sick'] + balance['compOff'] + balance['casual']
        
    elif leave_type == 'compOff':
        classification = classify_reason(reason) if reason else 'casual'
        
        if classification == 'sick':
            # Comp off with SICK reason: Comp Off → Sick → Casual
            if balance['compOff'] > 0:
                available_types.append(('Comp Off', balance['compOff']))
            if balance['sick'] > 0:
                available_types.append(('Sick Leave', balance['sick']))
            if balance['casual'] > 0:
                available_types.append(('Casual Leave', balance['casual']))
            total_available = balance['compOff'] + balance['sick'] + balance['casual']
        else:
            # Comp off with CASUAL reason: Comp Off → Casual
            if balance['compOff'] > 0:
                available_types.append(('Comp Off', balance['compOff']))
            if balance['casual'] > 0:
                available_types.append(('Casual Leave', balance['casual']))
            total_available = balance['compOff'] + balance['casual']
    
    balance_text = "\n".join([f"- {leave_type}: {avail_days} day(s)" for leave_type, avail_days in available_types])
    lop_amount = calculate_lop_amount(emp_email, balance['lop'])
    
    if total_available > 0:
        shortfall = max(0, days - total_available)
        
        if shortfall > 0:
            additional_lop = calculate_lop_amount(emp_email, shortfall)
            response_text = f'''⚠️ You have some leave balance available, but partial LOP is required:

{balance_text}

Current LOP Days: {balance['lop']} days (₹{lop_amount})

This will use all available balance + {shortfall} LOP day(s) (₹{additional_lop})
Requires HR approval (LOP is never auto-approved).

Would you like to proceed?'''
            
            session['pending_approval'] = True
            session['needs_review'] = True
            session['review_type'] = 'EMERGENCY' if is_emergency else 'NORMAL'
            
            return {
                'response': response_text,
                'buttons': [
                    {'text': 'Submit for HR Approval', 'action': 'submitForApproval'},
                    {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision'
            }
        else:
            response_text = f'''You have sufficient leave balance available:

{balance_text}

Current LOP Days: {balance['lop']} days (₹{lop_amount})

This will use all available balance to cover {days} day(s).

Would you like to proceed?'''
            
            return {
                'response': response_text,
                'buttons': [
                    {'text': 'Submit for Approval', 'action': 'submitForApproval'},
                    {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision'
            }
    
    # No balances available - show full LOP with amount
    return proceed_to_lop_db(emp_email, leave_data, balance, is_emergency)

def proceed_to_lop_db(emp_email, leave_data, balance, is_emergency=False):
    """Handle full LOP case - NO balance at all"""
    days = leave_data.get('duration', 0)
    lop_amount = calculate_lop_amount(emp_email, days)
    current_lop_amount = calculate_lop_amount(emp_email, balance['lop'])
    
    recipient = "HR" if is_emergency else "Manager/HR"
    status = 0 if is_emergency else 4
    
    session['pending_approval'] = True
    session['needs_review'] = True
    session['review_type'] = 'EMERGENCY' if is_emergency else 'NORMAL'
    
    return {
        'response': f'''⚠️ You don't have sufficient leave balance for any leave type.

Current balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{current_lop_amount})

This will affect your KPI and ₹{lop_amount} LOP will be deducted from your salary for {days} day(s).
This request requires {recipient} approval (LOP is never auto-approved).

Do you want to proceed to {recipient} review?''',
        'buttons': [
            {'text': f'Submit for {recipient} Approval', 'action': 'submitForApproval'},
            {'text': 'Cancel Leave', 'action': 'cancelLeave'}
        ],
        'state': 'waitingDecision'
    }

# ===================== FLASK APP =====================

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Initialize session data
def init_session():
    # Get balance from database for the employee INCLUDING LOP
    balance = get_employee_balance(EMPLOYEE_EMAIL)
    session['leave_balance'] = balance
    
    # Performance tracking variables (hardcoded as per requirements)
    if 'last_month_attendance' not in session:
        session['last_month_attendance'] = 95.0
    if 'last_month_performance' not in session:
        session['last_month_performance'] = 95.0
    if 'late_minutes_used' not in session:
        session['late_minutes_used'] = 0
    if 'last_3_months_attendance' not in session:
        session['last_3_months_attendance'] = 90.0
    if 'leave_pattern_violation' not in session:
        session['leave_pattern_violation'] = False
    if 'kpi_remarks' not in session:
        session['kpi_remarks'] = []
    
    if 'employee_salary' not in session:
        session['employee_salary'] = 30000
    
    # Add LOP to session
    if 'lop_amount' not in session:
        session['lop_amount'] = calculate_lop_amount(EMPLOYEE_EMAIL)
    
    session.modified = True

# HTML Template (same as original)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leave Management Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 90%;
            max-width: 600px;
            height: 80vh;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 {
            font-size: 24px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .bot-message {
            text-align: left;
        }
        .user-message {
            text-align: right;
        }
        .message-content {
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .bot-message .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1)
        }
        .user-message .message-content {
            background: #667eea;
            color: white;
        }
        .button-group {
            margin-top: 10px;
        }
        .chat-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        .chat-button:hover {
            background: #5568d3;
        }
        .input-container {
            padding: 15px;
            background: white;
            border-top: 1px solid #ddd;
            display: none;
        }
        .input-container.active {
            display: flex;
            gap: 10px;
        }
        .input-container input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
        }
        .input-container button {
            padding: 12px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
        }
        .input-container button:hover {
            background: #5568d3;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🗓️ Leave Management Chatbot</h1>
            <p>Your smart assistant for leave applications</p>
        </div>
        <div class="chat-messages" id="chatMessages"></div>
        <div class="input-container" id="inputContainer">
            <input type="text" id="userInput" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let chatState = 'initial';
        let leaveData = {};

        function addMessage(text, type, buttons = null) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = text.replace(/\\n/g, '<br>');
            
            messageDiv.appendChild(contentDiv);
            
            if (buttons && buttons.length > 0) {
                const buttonGroup = document.createElement('div');
                buttonGroup.className = 'button-group';
                buttons.forEach(btn => {
                    const button = document.createElement('button');
                    button.className = 'chat-button';
                    button.textContent = btn.text;
                    button.onclick = () => handleButtonClick(btn.action, btn.text);
                    buttonGroup.appendChild(button);
                });
                contentDiv.appendChild(buttonGroup);
            }
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function showInput() {
            document.getElementById('inputContainer').classList.add('active');
            document.getElementById('userInput').focus();
        }

        function hideInput() {
            document.getElementById('inputContainer').classList.remove('active');
        }

        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            fetch('/process_input', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message, state: chatState, leaveData: leaveData})
            })
            .then(res => res.json())
            .then(data => {
                chatState = data.state;
                leaveData = data.leaveData;
                
                setTimeout(() => {
                    addMessage(data.response, 'bot', data.buttons);
                    
                    if (data.needsInput) {
                        showInput();
                    } else {
                        hideInput();
                    }
                }, 500);
            });
        }

        function handleButtonClick(action, text) {
            addMessage(text, 'user');
            
            fetch('/handle_action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: action, leaveData: leaveData})
            })
            .then(res => res.json())
            .then(data => {
                chatState = data.state;
                leaveData = data.leaveData;
                
                setTimeout(() => {
                    addMessage(data.response, 'bot', data.buttons);
                    
                    if (data.needsInput) {
                        showInput();
                    } else {
                        hideInput();
                    }
                }, 500);
            });
        }

        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Initialize chat
        window.onload = function() {
            fetch('/init')
            .then(res => res.json())
            .then(data => {
                addMessage(data.response, 'bot', data.buttons);
                chatState = data.state;
            });
        };
    </script>
</body>
</html>
'''

# Flask Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/init')
def init():
    init_session()
    balance = session['leave_balance']
    lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, balance['lop'])
    
    return jsonify({
        'response': f'''Welcome! 👋 How can I help you today?

Your current leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{lop_amount})''',
        'buttons': [
            {'text': 'Apply for a Leave', 'action': 'applyLeave'},
            {'text': 'Check Leave Balance', 'action': 'checkBalance'},
            {'text': 'Can I Apply for Leave?', 'action': 'canApply'},
            {'text': 'View Performance Metrics', 'action': 'viewMetrics'}
        ],
        'state': 'initial',
        'needsInput': False
    })

@app.route('/handle_action', methods=['POST'])
def handle_action():
    init_session()
    data = request.json
    action = data.get('action')
    leave_data = data.get('leaveData', {})
    
    # Get balance from database
    balance = get_employee_balance(EMPLOYEE_EMAIL)
    session['leave_balance'] = balance
    
    response = {'needsInput': False, 'buttons': None, 'state': 'initial', 'leaveData': leave_data}
    
    if action == 'applyLeave':
        response['response'] = 'Please select the type of leave you want to apply:'
        response['buttons'] = [
            {'text': 'Casual Leave or Sick Leave', 'action': 'casualOrSick'},
            {'text': 'Comp Off Leave', 'action': 'compOff'}
        ]
        response['state'] = 'selectLeaveType'
        
    elif action == 'checkBalance':
        lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, balance['lop'])
        response['response'] = f'''Your current leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{lop_amount})'''
        response['buttons'] = [
            {'text': 'Apply for a Leave', 'action': 'applyLeave'},
            {'text': 'Check Leave Balance', 'action': 'checkBalance'}
        ]
        
    elif action == 'canApply':
        has_leaves = balance['casual'] > 0 or balance['sick'] > 0 or balance['compOff'] > 0
        lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, balance['lop'])
        
        if has_leaves:
            response['response'] = f'''Yes, you can apply for leave! You have:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{lop_amount})'''
        else:
            response['response'] = f'''You currently have no leave balance. Applying for leave will result in Loss of Pay (LOP) and may affect your KPI.

Current LOP Days: {balance['lop']} days (₹{lop_amount})'''
        response['buttons'] = [
            {'text': 'Yes, Apply Leave', 'action': 'applyLeave'},
            {'text': 'No, Thanks', 'action': 'end'}
        ]
        
    elif action == 'casualOrSick':
        response['response'] = '''Please explain why you need leave.

Examples of valid reasons:
- "I am not feeling well"
- "I have fever"
- "Family function"
- "Personal work"
- "My mother is sick"

NOTE : Do not mention personal issue,personal work or personal problem as reason

Please enter your reason:'''
        response['needsInput'] = True
        response['state'] = 'enterReason'
        
    elif action == 'compOff':
        leave_data['leaveType'] = 'compOff'
        response['response'] = '''Please explain why you need leave.

Examples of valid reasons:
- "I am not feeling well"
- "I have fever"
- "Family function"
- "Personal work"

NOTE : Do not mention personal issue,personal work or personal problem as reason

Please enter your reason:'''
        response['needsInput'] = True
        response['state'] = 'enterReasonForCompOff'
        response['leaveData'] = leave_data
    
    elif action == 'viewMetrics':
        kpi_remarks = session.get('kpi_remarks', [])
        remarks_text = ""
        if kpi_remarks:
            remarks_text = "\n\n📝 Recent KPI Remarks:"
            for remark in kpi_remarks[-3:]:
                remarks_text += f"\n• {remark['remark']} ({remark['timestamp']})"
        
        # Get monthly counters from database
        current_month = datetime.now().strftime('%Y-%m')
        current_month_approved = get_monthly_auto_approved_days(EMPLOYEE_EMAIL, current_month)
        current_month_emergency = get_monthly_emergency_requests(EMPLOYEE_EMAIL, current_month)
        
        next_month = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).strftime('%Y-%m')
        next_month_approved = get_monthly_auto_approved_days(EMPLOYEE_EMAIL, next_month)
        next_month_emergency = get_monthly_emergency_requests(EMPLOYEE_EMAIL, next_month)
        
        current_month_name = datetime.now().strftime('%B %Y')
        next_month_name = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).strftime('%B %Y')
        
        lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, balance['lop'])
        
        metrics = f'''📊 Your Performance Metrics:

📅 Last Month Attendance: {session.get('last_month_attendance', 95.0)}%
💼 Last Month Performance: {session.get('last_month_performance', 95.0)}%
⏰ Late Minutes Used: {session.get('late_minutes_used', 0)}/120 minutes
📈 Last 3 Months Avg Attendance: {session.get('last_3_months_attendance', 75.0)}%

💰 Current LOP Days: {balance['lop']} days (₹{lop_amount})

📆 Monthly Leave Tracking:
{current_month_name}:
✅ Auto-approved days used: {current_month_approved}/2
🚨 Emergency requests used: {current_month_emergency}/1

{next_month_name}:
✅ Auto-approved days used: {next_month_approved}/2
🚨 Emergency requests used: {next_month_emergency}/1{remarks_text}'''
        
        is_eligible, violations = check_leave_eligibility()
        
        if is_eligible:
            metrics += "\n\n✅ You are eligible for automatic leave approval!"
        else:
            metrics += "\n\n⚠️ You are NOT eligible for automatic leave approval due to:"
            for violation in violations:
                metrics += f"\n{violation}"
        
        response['response'] = metrics
        response['buttons'] = [
            {'text': 'Apply for a Leave', 'action': 'applyLeave'},
            {'text': 'Check Leave Balance', 'action': 'checkBalance'}
        ]

    elif action == 'applyAsCasual':
        leave_data['leaveType'] = 'casual'
        response['response'] = 'Please enter the FROM date (YYYY-MM-DD):'
        response['needsInput'] = True
        response['state'] = 'enterFromDate'
        response['leaveData'] = leave_data
        
    elif action == 'applyAsSick':
        leave_data['leaveType'] = 'sick'
        response['response'] = 'Please enter the FROM date (YYYY-MM-DD):'
        response['needsInput'] = True
        response['state'] = 'enterFromDate'
        response['leaveData'] = leave_data
        
    elif action == 'applyAsCompOff':
        leave_data['leaveType'] = 'compOff'
        response['response'] = 'Please enter the FROM date (YYYY-MM-DD):'
        response['needsInput'] = True
        response['state'] = 'enterFromDate'
        response['leaveData'] = leave_data
        
    elif action == 'switchToCasual':
        leave_data['leaveType'] = 'casual'
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'switchToSick':
        leave_data['leaveType'] = 'sick'
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'switchToCompoff':
        leave_data['leaveType'] = 'compOff'
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'useSickOnly':
        # Use sick leave only - use the main processing function
        leave_data['leaveType'] = 'sick'
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'useCasualOnly':
        # Use casual leave only - use the main processing function
        leave_data['leaveType'] = 'casual'
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'useCompOffOnly':
        # Use comp off only - use the main processing function
        leave_data['leaveType'] = 'compOff'
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'usePartialCasualWithCompOff':
        # Process partial combination - use the main processing function
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
    
    elif action == 'useCombinationForSick':
        # Process sick with combination - use the main processing function
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
    
    elif action == 'useCasualCompOffForSick':
        # Process sick without sick balance - use the main processing function
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        
    elif action == 'submitRemainingForApproval':
        # Submit remaining days for HR/Manager approval
        remaining_data = session.get('remaining_leave_data', {})
        remaining_days = session.get('remaining_days', 0)
        already_approved = session.get('already_approved_days', 0)
        is_emergency = session.get('is_emergency_leave', False)
        
        print(f"\n{'='*60}")
        print(f"🔍 SUBMIT REMAINING FOR APPROVAL:")
        print(f"   Remaining Data: {remaining_data}")
        print(f"   Remaining Days: {remaining_days}")
        print(f"   Already Approved: {already_approved}")
        print(f"   Is Emergency: {is_emergency}")
        print(f"{'='*60}\n")
        
        if not remaining_data or remaining_days == 0:
            response['response'] = '❌ No remaining leave to submit. Starting over...'
            response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
            response['state'] = 'initial'
            session.pop('remaining_leave_data', None)
            session.pop('remaining_days', None)
            session.pop('already_approved_days', None)
            session.pop('is_emergency_leave', None)
            session.modified = True
            return jsonify(response)
        
        # Determine recipient and status
        if is_emergency:
            recipient = "HR"
            status = 0  # HR Review
        else:
            recipient = "Manager/HR"
            status = 4  # Manager Review
        
        # ✅ CRITICAL FIX: Insert Record 2 with SAME leave_data (same dates, same duration)
        # Only difference is the STATUS field
        leave_id = insert_leave_record(remaining_data, EMPLOYEE_EMAIL, status, approved_days=0)
        
        if leave_id:
            # Calculate what balance will be used if HR approves
            balance = get_employee_balance(EMPLOYEE_EMAIL)
            balance = {
                'casual': float(balance.get('casual', 0)),
                'sick': float(balance.get('sick', 0)),
                'compOff': float(balance.get('compOff', 0)),
                'lop': float(balance.get('lop', 0))
            }
            leave_type = remaining_data.get('leaveType')
            reason = remaining_data.get('reason', '')
            
            hr_balance_usage = []
            hr_lop_days = 0
            remaining_to_use = float(remaining_days)  # ✅ FIX

            if leave_type == 'casual':
                # Use casual first, then comp-off
                casual_avail = max(0, float(balance['casual']))  # ✅ FIX
                casual_used = min(casual_avail, remaining_to_use)
                if casual_used > 0:
                    hr_balance_usage.append(f"{casual_used} Casual Leave")
                    remaining_to_use -= casual_used
                
                if remaining_to_use > 0:
                    compoff_avail = max(0, float(balance['compOff']))  # ✅ FIX
                    compoff_used = min(compoff_avail, remaining_to_use)
                    if compoff_used > 0:
                        hr_balance_usage.append(f"{compoff_used} Comp Off")
                        remaining_to_use -= compoff_used
                
                if remaining_to_use > 0:
                    hr_lop_days = remaining_to_use
                    
            elif leave_type == 'sick':
                # Use sick first, then comp-off, then casual
                sick_avail = max(0, float(balance['sick']))  # ✅ FIX
                sick_used = min(sick_avail, remaining_to_use)
                if sick_used > 0:
                    hr_balance_usage.append(f"{sick_used} Sick Leave")
                    remaining_to_use -= sick_used
                
                if remaining_to_use > 0:
                    compoff_avail = max(0, float(balance['compOff']))  # ✅ FIX
                    compoff_used = min(compoff_avail, remaining_to_use)
                    if compoff_used > 0:
                        hr_balance_usage.append(f"{compoff_used} Comp Off")
                        remaining_to_use -= compoff_used
                
                if remaining_to_use > 0:
                    casual_avail = max(0, float(balance['casual']))  # ✅ FIX
                    casual_used = min(casual_avail, remaining_to_use)
                    if casual_used > 0:
                        hr_balance_usage.append(f"{casual_used} Casual Leave")
                        remaining_to_use -= casual_used
                
                if remaining_to_use > 0:
                    hr_lop_days = remaining_to_use
                    
            elif leave_type == 'compOff':
                # Check reason classification
                classification = classify_reason(reason) if reason else 'casual'
                
                if classification == 'sick':
                    # Comp off with sick reason: compOff → sick → casual
                    compoff_avail = max(0, float(balance['compOff']))  # ✅ FIX
                    compoff_used = min(compoff_avail, remaining_to_use)
                    if compoff_used > 0:
                        hr_balance_usage.append(f"{compoff_used} Comp Off")
                        remaining_to_use -= compoff_used
                    
                    if remaining_to_use > 0:
                        sick_avail = max(0, float(balance['sick']))  # ✅ FIX
                        sick_used = min(sick_avail, remaining_to_use)
                        if sick_used > 0:
                            hr_balance_usage.append(f"{sick_used} Sick Leave")
                            remaining_to_use -= sick_used
                    
                    if remaining_to_use > 0:
                        casual_avail = max(0, float(balance['casual']))  # ✅ FIX
                        casual_used = min(casual_avail, remaining_to_use)
                        if casual_used > 0:
                            hr_balance_usage.append(f"{casual_used} Casual Leave")
                            remaining_to_use -= casual_used
                else:
                    # Comp off with casual reason: compOff → casual
                    compoff_avail = max(0, float(balance['compOff']))  # ✅ FIX
                    compoff_used = min(compoff_avail, remaining_to_use)
                    if compoff_used > 0:
                        hr_balance_usage.append(f"{compoff_used} Comp Off")
                        remaining_to_use -= compoff_used
                    
                    if remaining_to_use > 0:
                        casual_avail = max(0, float(balance['casual']))  # ✅ FIX
                        casual_used = min(casual_avail, remaining_to_use)
                        if casual_used > 0:
                            hr_balance_usage.append(f"{casual_used} Casual Leave")
                            remaining_to_use -= casual_used
                
                if remaining_to_use > 0:
                    hr_lop_days = remaining_to_use
            
            # Build HR balance usage text
            if hr_balance_usage:
                hr_usage_text = " + ".join(hr_balance_usage)
                if hr_lop_days > 0:
                    lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, hr_lop_days)
                    hr_usage_text += f" + {hr_lop_days} LOP (₹{lop_amount})"
            else:
                if hr_lop_days > 0:
                    lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, hr_lop_days)
                    hr_usage_text = f"{hr_lop_days} LOP (₹{lop_amount})"
                else:
                    hr_usage_text = "available balance"
            
            lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, balance['lop'])
            response['response'] = f'''✅ Your remaining {remaining_days} day(s) have been submitted to {recipient} for approval!

Leave Application Summary:
✅ {already_approved} day(s) - Auto-approved
⏳ {remaining_days} day(s) - Pending {recipient} approval

If {recipient} approves, it will use: {hr_usage_text}

Your current leave balance (unchanged until {recipient} approves):
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{lop_amount})'''
        else:
            response['response'] = '❌ Error submitting remaining leave. Please try again.'
        
        # Clear session data
        session.pop('remaining_leave_data', None)
        session.pop('remaining_days', None)
        session.pop('already_approved_days', None)
        session.pop('is_emergency_leave', None)
        session.modified = True
        
        response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
        response['state'] = 'initial'
        response['leaveData'] = {}
        
    elif action == 'proceedWithEmergencyPartial':
        # Process emergency with only 1 day auto-approval due to eligibility
        leave_data['isEmergency'] = True
        
        # Manually set approved_days = 1, hr_days = remaining
        days = leave_data.get('duration', 0)
        approved_days = 1
        hr_days = days - 1
        
        # Get balance
        balance = get_employee_balance(EMPLOYEE_EMAIL)
        balance = {
            'casual': float(balance.get('casual', 0)),
            'sick': float(balance.get('sick', 0)),
            'compOff': float(balance.get('compOff', 0)),
            'lop': float(balance.get('lop', 0))
        }
        
        # Auto-approve 1 day with LOP logic
        new_balance = balance.copy()
        leave_type = leave_data.get('leaveType')
        
        lop_to_increment = 0.0
        used_parts = []
        
        if leave_type == 'casual':
            if new_balance['casual'] >= 1:
                new_balance['casual'] -= 1
                used_parts.append(f"1 Casual Leave")
            elif new_balance['compOff'] >= 1:
                new_balance['compOff'] -= 1
                used_parts.append(f"1 Comp Off")
            else:
                lop_to_increment = 1
                used_parts.append(f"1 LOP")
        
        # Ensure no negative balances
        new_balance['casual'] = max(0, new_balance['casual'])
        new_balance['sick'] = max(0, new_balance['sick'])
        new_balance['compOff'] = max(0, new_balance['compOff'])
        new_balance['lop'] += lop_to_increment
        
        # Update database
        update_employee_balance(EMPLOYEE_EMAIL, new_balance['casual'], 
                              new_balance['sick'], new_balance['compOff'],
                              new_balance['lop'])
        
        # Insert auto-approved record
        leave_id = insert_leave_record(leave_data, EMPLOYEE_EMAIL, 1, 'EMERGENCY', 
                                     approved_days=1.0, lop_days=lop_to_increment)
        
        if leave_id:
            update_leave_status_db(leave_id, 1, "Emergency auto-approved (1 day - eligibility override)", "SYSTEM", 'EMERGENCY')
            
            # Store remaining for HR
            session['remaining_leave_data'] = leave_data.copy()
            session['remaining_days'] = float(hr_days)
            session['already_approved_days'] = 1.0
            session['is_emergency_leave'] = True
            session.modified = True
            
            # Get monthly counters
            from_date_str = leave_data.get('fromDate')
            if from_date_str:
                try:
                    from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
                    leave_month = from_date.strftime('%Y-%m')
                except:
                    leave_month = datetime.now().strftime('%Y-%m')
            else:
                leave_month = datetime.now().strftime('%Y-%m')
            
            monthly_approved = get_monthly_auto_approved_days(EMPLOYEE_EMAIL, leave_month)
            monthly_emergency = get_monthly_emergency_requests(EMPLOYEE_EMAIL, leave_month)
            
            used_type = " + ".join(used_parts) if used_parts else "1 day"
            lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, lop_to_increment)
            
            response['response'] = f'''✅ 1 day has been auto-approved using {used_type} (Emergency eligibility override)!

Since you have eligibility violations, only 1 day can be auto-approved for emergency.

The remaining {hr_days} day(s) require HR approval.

Your updated leave balance (after auto-approval):
📅 Casual Leave: {new_balance['casual']} days
🏥 Sick Leave: {new_balance['sick']} days
⏰ Comp Off: {new_balance['compOff']} days
💰 LOP Days: {new_balance['lop']} days (₹{lop_amount})

📊 Monthly auto-approved days used for {leave_month}: {float(monthly_approved) + 1}/2
🚨 Emergency requests used for {leave_month}: {int(monthly_emergency) + 1}/1

Do you want to submit the remaining {hr_days} day(s) for HR approval?'''
            
            response['buttons'] = [
                {'text': 'Submit Remaining for HR Approval', 'action': 'submitRemainingForApproval'},
                {'text': 'Cancel Remaining Leave', 'action': 'cancelLeave'}
            ]
            response['state'] = 'waitingDecision'
            response['leaveData'] = leave_data
        else:
            response['response'] = '❌ Error processing emergency approval.'
            response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
            response['state'] = 'initial'
            response['leaveData'] = {}
                
    elif action == 'submitForApproval':
        # Submit for HR/Manager approval
        leave_type = leave_data.get('leaveType')
        days = leave_data.get('duration', 0)
        reason = leave_data.get('reason', '')
        from_date_str = leave_data.get('fromDate')
        
        # Check if emergency (from session or re-evaluate)
        is_emergency = leave_data.get('isEmergency', False)
        if not is_emergency and from_date_str:
            is_emergency = is_emergency_leave_ai(reason, from_date_str)
        
        if is_emergency:
            recipient = "HR"
            status = 0  # HR Review
        else:
            recipient = "Manager/HR"
            status = 4  # Manager Review
        
        # Calculate what would be used if approved based on PRIORITY ORDER
        parts = []
        remaining = days
        
        if leave_type == 'casual':
            # Casual: Casual → Comp Off
            casual_used = min(balance['casual'], remaining)
            if casual_used > 0:
                parts.append(f"{casual_used} Casual Leave")
                remaining -= casual_used
            
            if remaining > 0:
                compoff_used = min(balance['compOff'], remaining)
                if compoff_used > 0:
                    parts.append(f"{compoff_used} Comp Off")
                    remaining -= compoff_used
        
        elif leave_type == 'sick':
            # Sick: Sick → Comp Off → Casual
            sick_used = min(balance['sick'], remaining)
            if sick_used > 0:
                parts.append(f"{sick_used} Sick Leave")
                remaining -= sick_used
            
            if remaining > 0:
                compoff_used = min(balance['compOff'], remaining)
                if compoff_used > 0:
                    parts.append(f"{compoff_used} Comp Off")
                    remaining -= compoff_used
            
            if remaining > 0:
                casual_used = min(balance['casual'], remaining)
                if casual_used > 0:
                    parts.append(f"{casual_used} Casual Leave")
                    remaining -= casual_used
        
        elif leave_type == 'compOff':
            # Comp Off: Check reason
            classification = classify_reason(reason) if reason else 'casual'
            
            if classification == 'sick':
                # Comp Off with SICK reason: Comp Off → Sick → Casual
                compoff_used = min(balance['compOff'], remaining)
                if compoff_used > 0:
                    parts.append(f"{compoff_used} Comp Off")
                    remaining -= compoff_used
                
                if remaining > 0:
                    sick_used = min(balance['sick'], remaining)
                    if sick_used > 0:
                        parts.append(f"{sick_used} Sick Leave")
                        remaining -= sick_used
                
                if remaining > 0:
                    casual_used = min(balance['casual'], remaining)
                    if casual_used > 0:
                        parts.append(f"{casual_used} Casual Leave")
                        remaining -= casual_used
            else:
                # Comp Off with CASUAL reason: Comp Off → Casual
                compoff_used = min(balance['compOff'], remaining)
                if compoff_used > 0:
                    parts.append(f"{compoff_used} Comp Off")
                    remaining -= compoff_used
                
                if remaining > 0:
                    casual_used = min(balance['casual'], remaining)
                    if casual_used > 0:
                        parts.append(f"{casual_used} Casual Leave")
                        remaining -= casual_used
        
        # Add LOP if any remaining days
        if remaining > 0:
            lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, remaining)
            parts.append(f"{remaining} LOP (₹{lop_amount})")
        
        leave_source = " + ".join(parts) if parts else f"{days} LOP (₹{calculate_lop_amount(EMPLOYEE_EMAIL, days)})"
        
        # Insert leave application (ONLY ONCE - at final submission)
        leave_id = insert_leave_record(leave_data, EMPLOYEE_EMAIL, status, approved_days=0)
        
        if leave_id:
            lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, balance['lop'])
            response['response'] = f'''📤 Your leave application has been submitted and is waiting for {recipient} approval.

Application Details:
- Leave Type: {'Comp Off' if leave_type == 'compOff' else leave_type.title() + ' Leave'}
- Duration: {days} day(s)
- Reason: {reason}

If approved, it will use: {leave_source}

Your leave balance remains unchanged pending approval:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{lop_amount})'''
        else:
            response['response'] = '❌ Error submitting leave application. Please try again.'
        
        response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
        response['state'] = 'initial'
        response['leaveData'] = {}
        
    elif action == 'cancelLeave':
        response['response'] = 'Leave application cancelled. Starting over...'
        response['buttons'] = [
            {'text': 'Apply for a Leave', 'action': 'applyLeave'},
            {'text': 'Check Leave Balance', 'action': 'checkBalance'},
            {'text': 'Can I Apply for Leave?', 'action': 'canApply'}
        ]
        response['leaveData'] = {}
        
    elif action == 'end':
        response['response'] = 'Thank you! Have a great day! 😊'
        response['buttons'] = [
            {'text': 'Start Again', 'action': 'startAgain'}
        ]
        
    elif action == 'startAgain':
        # Refresh balance from database
        balance = get_employee_balance(EMPLOYEE_EMAIL)
        session['leave_balance'] = balance
        
        # Clear any pending session data
        session.pop('calculated_balance', None)
        session.pop('pending_approval', None)
        session.pop('needs_review', None)
        session.pop('review_type', None)
        session.pop('emergency_no_balance', None)
        session.pop('remaining_leave_data', None)
        session.pop('remaining_days', None)
        session.pop('already_approved_days', None)
        
        lop_amount = calculate_lop_amount(EMPLOYEE_EMAIL, balance['lop'])
        
        response['response'] = f'''Welcome back! 👋 How can I help you today?

Your current leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days
💰 LOP Days: {balance['lop']} days (₹{lop_amount})'''
        response['buttons'] = [
            {'text': 'Apply for a Leave', 'action': 'applyLeave'},
            {'text': 'Check Leave Balance', 'action': 'checkBalance'},
            {'text': 'Can I Apply for Leave?', 'action': 'canApply'}
        ]
        response['leaveData'] = {}
        
    return jsonify(response)

@app.route('/process_input', methods=['POST'])
def process_input():
    init_session()
    data = request.json
    message = data.get('message')
    state = data.get('state')
    leave_data = data.get('leaveData', {})
    
    response = {'needsInput': False, 'buttons': None, 'state': 'initial', 'leaveData': leave_data}
    
    if state == 'enterReason':
        classification = classify_reason(message)
        
        if classification == "invalid":
            response['response'] = '''❌ Please provide a valid leave reason.

Your reason should clearly explain why you need leave. For example:
- "I am not feeling well" (for sick leave)
- "I have fever" (for sick leave)
- "Family function" (for casual leave)
- "Personal work" (for casual leave)
- "My mother is sick" (for casual leave)
- "Accident" (for emergency leave)
- "Sudden hospitalization" (for emergency leave)

NOTE: Do not mention personal issue, personal work or personal problem as reason

Please try again with a proper reason:'''
            response['needsInput'] = True
            response['state'] = 'enterReason'
            return jsonify(response)
        
        leave_data['reason'] = message
        
        is_sick_reason = classification == "sick"
        
        if classification == "sick":
            leave_data['leaveType'] = 'sick'
            response['response'] = f"I understand you're not feeling well. You are applying for Sick Leave.\n\nPlease enter the FROM date (YYYY-MM-DD):"
            response['needsInput'] = True
            response['state'] = 'enterFromDate'
            response['leaveData'] = leave_data
        else:
            leave_data['leaveType'] = 'casual'
            response['response'] = f'You are applying for Casual Leave.\n\nPlease enter the FROM date (YYYY-MM-DD):'
            response['needsInput'] = True
            response['state'] = 'enterFromDate'
            response['leaveData'] = leave_data
    
    elif state == 'enterReasonForCompOff':
        classification = classify_reason(message)
        
        if classification == "invalid":
            response['response'] = '''❌ Please provide a valid leave reason.

Your reason should clearly explain why you need leave. For example:
- "I am not feeling well" (for sick leave)
- "I have fever" (for sick leave)
- "Family function" (for casual leave)
- "Personal work" (for casual leave)

NOTE : Do not mention personal issue,personal work or personal problem as reason

Please try again with a proper reason:'''
            response['needsInput'] = True
            response['state'] = 'enterReasonForCompOff'
            return jsonify(response)
            
        leave_data['reason'] = message
        response['response'] = 'Please enter the FROM date (YYYY-MM-DD):'
        response['needsInput'] = True
        response['state'] = 'enterFromDate'
        response['leaveData'] = leave_data
            
    elif state == 'enterFromDate':
        if not is_valid_date_format(message):
            response['response'] = '❌ Invalid date format! Please enter date in YYYY-MM-DD format (e.g., 2024-12-25):'
            response['needsInput'] = True
            response['state'] = 'enterFromDate'
            return jsonify(response)
        
        leave_type = leave_data.get('leaveType', '')
        reason = leave_data.get('reason', '')
        
        # Check emergency BEFORE date validation
        is_emergency = False
        if leave_type != 'sick':  # Sick leaves cannot be emergency
            is_emergency = is_emergency_leave_ai(reason, message, leave_type)
        leave_data['isEmergency'] = is_emergency
        
        print(f"\n{'='*60}")
        print(f"📝 EMERGENCY DETECTION SUMMARY:")
        print(f"   Reason: {reason}")
        print(f"   From Date: {message}")
        print(f"   Leave Type: {leave_type}")
        print(f"   Is Emergency: {is_emergency}")
        print(f"{'='*60}\n")
        
        if leave_type == 'sick':
            if not is_valid_sick_leave_from_date(message):
                response['response'] = '❌ For sick leaves, FROM date can only be past dates, today, or tomorrow. Please enter a valid FROM date:'
                response['needsInput'] = True
                response['state'] = 'enterFromDate'
                return jsonify(response)
        
        elif leave_type == 'casual':
            if not validate_casual_leave_date(message, is_emergency):
                if is_emergency:
                    response['response'] = "❌ For EMERGENCY casual leave, FROM date MUST be TODAY. Please enter today's date:"
                else:
                    response['response'] = '❌ For normal casual leave, FROM date must be TOMORROW or later (one day prior rule). You CANNOT apply casual leave for TODAY. Please enter a valid FROM date:'
                response['needsInput'] = True
                response['state'] = 'enterFromDate'
                return jsonify(response)
        
        elif leave_type == 'compOff':
            if not validate_compoff_leave_date(message, is_emergency, reason):
                if is_emergency:
                    response['response'] = "❌ For EMERGENCY comp off, FROM date MUST be TODAY. Please enter today's date:"
                else:
                    # Check reason classification
                    classification = classify_reason(reason) if reason else 'casual'
                    if classification == 'sick':
                        response['response'] = '❌ For Comp Off with sick reason, FROM date can be past, today, or tomorrow. Please enter a valid FROM date:'
                    else:
                        response['response'] = '❌ For Comp Off with casual reason, FROM date must be TOMORROW or later (one day prior rule). You CANNOT apply comp off for TODAY. Please enter a valid FROM date:'
                response['needsInput'] = True
                response['state'] = 'enterFromDate'
                return jsonify(response)
    
        # Check for overlapping leaves
        overlapping_leaves = get_overlapping_leaves(EMPLOYEE_EMAIL, message, message)
        if overlapping_leaves:
            response['response'] = f'❌ You already have a leave application for {message}. Please choose a different date:'
            response['needsInput'] = True
            response['state'] = 'enterFromDate'
            return jsonify(response)
        
        leave_data['fromDate'] = message
                
        # Add emergency note if applicable
        if is_emergency:
            emergency_note = " 🚨 EMERGENCY DETECTED"
        else:
            emergency_note = ""
            
        response['response'] = f'Please enter the TO date (YYYY-MM-DD):{emergency_note}'
        response['needsInput'] = True
        response['state'] = 'enterToDate'
        response['leaveData'] = leave_data
                
    elif state == 'enterToDate':
        if not is_valid_date_format(message):
            response['response'] = '❌ Invalid date format! Please enter date in YYYY-MM-DD format (e.g., 2024-12-25):'
            response['needsInput'] = True
            response['state'] = 'enterToDate'
            return jsonify(response)
        
        leave_type = leave_data.get('leaveType', '')
        is_sick_reason = is_employee_sick(leave_data.get('reason', ''))
        from_date = leave_data.get('fromDate')
        
        if leave_type == 'sick' or is_sick_reason:
            if not is_valid_sick_leave_to_date(message, from_date):
                response['response'] = '❌ TO date cannot be before FROM date! Please enter a valid TO date:'
                response['needsInput'] = True
                response['state'] = 'enterToDate'
                return jsonify(response)
        elif leave_type == 'casual' or leave_type == 'compOff':
            if not is_valid_casual_comp_off_to_date(message, from_date):
                response['response'] = '❌ TO date cannot be before FROM date! Please enter a valid TO date:'
                response['needsInput'] = True
                response['state'] = 'enterToDate'
                return jsonify(response)
        
        # Check for overlapping leaves
        overlapping_leaves = get_overlapping_leaves(EMPLOYEE_EMAIL, from_date, message)
        if overlapping_leaves:
            response['response'] = '❌ These dates overlap with your already applied leaves! Please choose different dates:'
            response['needsInput'] = True
            response['state'] = 'enterToDate'
            return jsonify(response)
        start_date = datetime.strptime(from_date, '%Y-%m-%d')
        end_date = datetime.strptime(message, '%Y-%m-%d')
        holidays = get_holidays_in_range(from_date, message)
        working_days_check = 0
        all_sundays = True
        all_holidays = True
        holiday_names = []
        current = start_date
        while current <= end_date:
            is_sunday = current.weekday() == 6
            is_holiday = current.date() in holidays
            
            if not is_sunday:
                all_sundays = False
            if not is_holiday:
                all_holidays = False
                
            if is_holiday:
                # Get holiday name
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    query = "SELECT value FROM holiday WHERE date = %s AND status IS NULL"
                    cursor.execute(query, (current.strftime('%Y-%m-%d'),))
                    result = cursor.fetchone()
                    if result:
                        holiday_names.append(f"{current.strftime('%d %b')} - {result['value']}")
                    cursor.close()
                    conn.close()
            
            if not is_sunday and not is_holiday:
                working_days_check += 1
                
            current += timedelta(days=1)
        
        # Check if ALL days are Sundays
        if all_sundays and (end_date - start_date).days == 0:
            response['response'] = '''❌ You cannot apply for leave on Sunday only.
            
    Sundays are weekly offs and are not counted as working days.

    Please select a different date range that includes working days.'''
            response['needsInput'] = True
            response['state'] = 'enterToDate'
            return jsonify(response)
        
        # Check if ALL days are holidays
        if all_holidays and len(holiday_names) > 0:
            holiday_list_text = "\n".join([f"   • {h}" for h in holiday_names])
            response['response'] = f'''❌ You cannot apply for leave on holidays only.

    The following dates are holidays:
    {holiday_list_text}

    Holidays are not counted as working days.

    Please select a different date range that includes working days.'''
            response['needsInput'] = True
            response['state'] = 'enterToDate'
            return jsonify(response)
        
        # Check if date range has ZERO working days (mix of Sundays and holidays)
        if working_days_check == 0:
            response['response'] = '''❌ Your selected date range has no working days.

    All dates fall on Sundays and/or holidays, which are not counted as working days.

    Please select a different date range that includes at least one working day.'''
            response['needsInput'] = True
            response['state'] = 'enterToDate'
            return jsonify(response)
        leave_data['toDate'] = message
        days = calculate_days(leave_data['fromDate'], message)
        leave_data['duration'] = days
        
        # Calculate total calendar days for information
        start_date = datetime.strptime(leave_data['fromDate'], '%Y-%m-%d')
        end_date = datetime.strptime(message, '%Y-%m-%d')
        calendar_days = (end_date - start_date).days + 1
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        is_past_leave = end_date < today
        
        # Get exclusion info
        exclusion_info = get_excluded_days_info(leave_data['fromDate'], message)
        num_sundays = exclusion_info['sundays']
        num_holidays = exclusion_info['holidays']
        holiday_list = exclusion_info['holiday_list']
        
        # Build exclusion text
        exclusions = []
        if num_sundays > 0:
            exclusions.append(f"{num_sundays} Sunday(s)")
        if num_holidays > 0:
            exclusions.append(f"{num_holidays} holiday(s)")
        
        if exclusions:
            exclusion_text = " and ".join(exclusions)
            response_text = f"You are applying for {days} working day(s) out of {calendar_days} calendar day(s).\n\n📅 Date Range: {leave_data['fromDate']} to {message}\n🚫 Excluded: {exclusion_text}"
            
            # Show holiday names
            if holiday_list:
                response_text += "\n\n🎉 Holidays in your leave period:"
                for h_date, h_name in holiday_list:
                    response_text += f"\n   • {h_date.strftime('%d %b')} - {h_name}"
        else:
            response_text = f"You are applying for {days} day(s) of leave from {leave_data['fromDate']} to {message}."

        if is_past_leave:
            response_text += "\n\n⚠️ Note: This is a past leave application."

        response['response'] = response_text
        
        # Process the leave application (does NOT insert into database yet)
        result = process_leave_application(EMPLOYEE_EMAIL, leave_data)
        response.update(result)
        response['leaveData'] = leave_data
        
    return jsonify(response)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False,
        use_reloader=False
    )
