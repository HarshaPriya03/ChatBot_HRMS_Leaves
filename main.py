from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, render_template_string, request, jsonify, session
from datetime import datetime, timedelta
import secrets
import re

from sentence_transformers import SentenceTransformer, util
import numpy as np


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

def detect_reason_pattern(leave_data, days_window=60, threshold=4):
    """
    Enhanced pattern detection using semantic similarity.
    
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

    count = 1 
    try:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = today - timedelta(days=days_window)
        
        for entry in session.get('applied_leaves', []):
            past_reason = entry.get('reason', '') or ''
            past_cat = entry.get('category')
            
            if not past_cat and past_reason:
                past_cat = categorize_reason_for_pattern(past_reason)
            
            if past_cat != category:
                continue
                
            ts_str = entry.get('timestamp') or entry.get('fromDate')
            if not ts_str:
                continue
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d')
            except:
                continue
            if ts >= cutoff:
                count += 1
    except Exception as e:
        print(f"Error in pattern detection: {e}")
        return (False, category, 0)

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


def add_kpi_remark(employee_key, remark):
    """Store KPI remarks in session for tracing. employee_key unused for single-user session."""
    if 'kpi_remarks' not in session:
        session['kpi_remarks'] = []
    session['kpi_remarks'].append({
        'remark': remark,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    session.modified = True

def is_emergency_leave_ai(reason: str, from_date_str: str = None) -> bool:
    """
    Use AI to detect if reason qualifies as emergency with strict criteria.
    Returns False immediately if from_date is provided and not today (RULE 13).
    
    STRICT EMERGENCY CRITERIA:
    - Must be unforeseen/sudden
    - Requires IMMEDIATE action TODAY
    - Cannot be planned or scheduled
    """
    if not reason:
        return False
    
    
    if from_date_str:
        try:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            
            if from_date != today:
                return False
        except:
            
            pass
    
    reason_lower = reason.lower().strip()
    
   
    non_emergency_keywords = [
        'tomorrow', 'next week', 'next month', 'planning', 'scheduled',
        'appointment', 'routine', 'checkup', 'follow-up', 'regular',
        'function', 'wedding', 'marriage', 'ceremony', 'celebration',
        'travel', 'trip', 'vacation', 'tour', 'visit',
        'personal work', 'bank work', 'documentation', 'paperwork',
        'taking care', 'looking after', 'need to help'
    ]
    

    for keyword in non_emergency_keywords:
        if keyword in reason_lower:
            return False
    
    
    non_emergency_patterns = [
        r'i (?:want|need|would like) to',  
        r'going (?:to|for)', 
        r'have (?:to|an) (?:go|attend)', 
        r'(?:mother|father|grandmother|grandfather|wife|husband|son|daughter|friend|relative).{0,20}(?:sick|ill|unwell|not feeling well)',  # Family member sick (not critical)
    ]
    
    for pattern in non_emergency_patterns:
        if re.search(pattern, reason_lower):
            return False
    
    critical_keywords = [
        'accident', 'injured', 'fracture', 'broken',
        'hospitalized', 'admitted', 'icu', 'critical',
        'died', 'death', 'passed away', 'expired', 'funeral',
        'sudden', 'unexpected', 'emergency room', 'ambulance',
        'heart attack', 'stroke', 'seizure', 'collapsed'
    ]
    
    has_critical_keyword = any(keyword in reason_lower for keyword in critical_keywords)
    
    
    classification = classify_reason(reason)
    if classification == "sick":
        return False 
    
    
    prompt = f"""You are an EMERGENCY LEAVE VALIDATOR for a company leave management system.

STRICT EMERGENCY CRITERIA - ALL must be true:
1. UNFORESEEN/SUDDEN - Could NOT have been planned in advance
2. REQUIRES IMMEDIATE ACTION TODAY - Cannot wait until tomorrow
3. CRITICAL SITUATION - Serious consequences if not addressed immediately

TRUE EMERGENCIES (return YES):
✓ Accidents/injuries requiring immediate medical attention
✓ Sudden hospitalization (critical condition)
✓ Death in immediate family
✓ Natural disasters affecting home/family
✓ Fire or major property damage
✓ Medical emergencies (heart attack, stroke, severe trauma)
✓ Critical family emergency requiring immediate presence

NOT EMERGENCIES (return NO):
✗ "I am sick" / "I have fever" → Use sick leave, not emergency
✗ Routine doctor appointments
✗ Planned surgeries or medical procedures
✗ Family functions, weddings, ceremonies
✗ "My [family] is sick" → Unless critical hospitalization
✗ Taking care of sick family member → Unless sudden critical condition
✗ Personal work, bank work, documentation
✗ Travel plans, trips, vacations
✗ Any reason that was planned/scheduled in advance
✗ Reasons with keywords "urgent" or "emergency" but not truly critical
✗ "Urgent personal work" → Not emergency
✗ "Emergency family function" → Not emergency

KEYWORD ABUSE CHECK:
If the reason just contains words like "urgent", "emergency", "important" but doesn't describe a CRITICAL UNFORESEEN situation, return NO.

Examples:
"Accident - broken leg" → YES (unforeseen, immediate)
"Father sudden heart attack hospitalized" → YES (critical, immediate)
"Death of grandmother" → YES (unforeseen, immediate)
"Urgent personal work" → NO (vague, not critical)
"Emergency - need to attend wedding" → NO (planned event, not critical)
"My mother is sick, need to take care" → NO (not critical hospitalization)
"I have fever and emergency" → NO (use sick leave)
"Urgent doctor appointment" → NO (planned appointment)
"Emergency family function" → NO (function = planned, not emergency)

Reason: "{reason}"

CRITICAL ANALYSIS:
1. Is this TRULY unforeseen (couldn't be planned)?
2. Does it require action TODAY (can't wait)?
3. Is there a CRITICAL situation with serious consequences?

If ALL THREE are YES, respond with: YES
Otherwise, respond with: NO

Your answer (YES or NO only):"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        outputs = model.generate(**inputs, max_length=10, temperature=0.0, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
        
        if "YES" in result and not has_critical_keyword:
            
            stricter_check = any(word in reason_lower for word in [
                'accident', 'hospitalized', 'death', 'died', 'critical', 
                'fracture', 'injured', 'heart attack', 'stroke'
            ])
            
            if not stricter_check:
                return False
        
        return "YES" in result
        
    except Exception as e:
        print(f"Error in emergency detection: {e}")
        return has_critical_keyword

# ----------------- DATE VALIDATION FUNCTIONS -----------------

def is_valid_casual_comp_off_from_date(date_string, is_emergency=False):
    """
    Check if FROM date is valid for Casual/Comp Off leave.
    
    Rules:
    - Emergency: FROM date MUST be TODAY only
    - Normal Casual: FROM date must be tomorrow or future (one day prior rule)
    - Comp Off: FROM date must be today or future
    """
    try:
        input_date = datetime.strptime(date_string, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if is_emergency:
            # Emergency MUST be today only
            return input_date == today
        else:
            # Normal casual: tomorrow or future (one day prior rule)
            # Comp off: today or future
            return input_date >= today
            
    except ValueError:
        return False

def validate_casual_leave_date(date_string, is_emergency=False):
    """
    Specific validation for casual leave with one-day-prior rule.
    
    Rules:
    - Emergency casual: FROM date MUST be TODAY
    - Normal casual: FROM date must be TOMORROW or future (applied one day prior)
    """
    try:
        input_date = datetime.strptime(date_string, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        if is_emergency:
            return input_date == today
        else:
            # Normal casual must be tomorrow or later (one day prior rule)
            return input_date >= tomorrow
            
    except ValueError:
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

# ----------------- LEAVE PROCESSING FUNCTIONS -----------------

def check_auto_approval_eligibility(leave_data, balance, is_emergency=False):
    """
    Central function to check if leave can be auto-approved.
    FIXED: Now checks quota for the month of leave application, not current month.
    
    Returns: (can_auto_approve: bool, reason: str, approved_days: int, hr_days: int)
    """
    days = leave_data.get('duration', 0)
    leave_type = leave_data.get('leaveType')
    from_date_str = leave_data.get('fromDate')
    
    # FIXED: Get counters for the leave month, not current month
    monthly_approved, monthly_emergency, leave_month = get_monthly_counters_for_leave(leave_data)
    
    # ========== CALCULATE AVAILABLE BALANCE FOR EACH LEAVE TYPE ==========
    if leave_type == 'casual':
        total_balance = balance.get('casual', 0) + balance.get('compOff', 0)
        available_for_type = total_balance
    elif leave_type == 'sick':
        sick_balance = balance.get('sick', 0)
        casual_balance = balance.get('casual', 0)
        comp_off_balance = balance.get('compOff', 0)
        total_balance = sick_balance + casual_balance + comp_off_balance
        available_for_type = total_balance
    elif leave_type == 'compOff':
        total_balance = balance.get('compOff', 0)
        available_for_type = total_balance
    else:
        total_balance = 0
        available_for_type = 0
    
    # ========== RULE 19: LOP is NEVER auto-approved ==========
    if total_balance == 0 and days > 0:
        if not is_emergency:
            return (False, "No leave balance available (LOP cannot be auto-approved)", 0, days)
    
    # ========== CHECK MONTHLY AUTO-APPROVAL LIMITS ==========
    remaining_monthly = 2 - monthly_approved
    
    # ========== EMERGENCY LEAVE LOGIC ==========
    if is_emergency:
        # RULE 9-10: Only 1 emergency request per month
        if monthly_emergency >= 1:
            return (False, f"Emergency quota exhausted for {leave_month} (1 per month)", 0, days)
        
        # RULE 12-13: Emergency FROM date must be TODAY
        if from_date_str:
            try:
                from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                if from_date != today:
                    return (False, "Emergency leave FROM date must be TODAY", 0, days)
            except:
                return (False, "Invalid FROM date format", 0, days)
        
        # RULE 20: Emergency with LB=0 can auto-approve 1 day
        if total_balance == 0:
            if days == 1:
                return (True, "Emergency override with no balance (1 day)", 1, 0)
            elif days > 1:
                return (True, "Emergency override (1 day approved, rest to HR)", 1, days - 1)
        
        # RULE 14-17: Emergency Auto-Approval Logic
        if remaining_monthly > 0:
            approved = min(days, remaining_monthly, available_for_type)
            hr_review = days - approved
            return (True, f"Emergency auto-approved ({approved} days within monthly limit)", approved, hr_review)
        else:
            approved = min(1, available_for_type)
            hr_review = days - approved
            return (True, "Emergency override (1 day approved)", approved, hr_review)
    
    # ========== NORMAL LEAVE LOGIC ==========
    else:
        # RULE 3: Once 2 days are auto-approved IN THAT MONTH, no further normal leave auto-approval
        if monthly_approved >= 2:
            return (False, f"Monthly auto-approval limit exhausted for {leave_month} (2 days)", 0, days)
        
        # RULE 4-5: Check eligibility violations
        is_eligible, violations = check_leave_eligibility()
        if not is_eligible:
            return (False, "Eligibility violations: " + "; ".join(violations), 0, days)
        
        # RULE 24-26: Check pattern abuse using SEMANTIC pattern detection
        pattern_detected, pattern_category, pattern_count = detect_reason_pattern(
            leave_data, days_window=60, threshold=3
        )
        if pattern_detected:
            add_kpi_remark(None, f"Semantic pattern abuse detected: {pattern_category} ({pattern_count} times)")
            return (False, f"Pattern abuse detected ({pattern_category} - {pattern_count} occurrences)", 0, days)
        
        # RULE 22: Normal leave auto-approval for 1 or 2 days
        # RULE 23: For 3+ days, partial auto-approval possible
        if days <= 2:
            if available_for_type >= days:
                approved = min(days, remaining_monthly)
                hr_review = days - approved
                if hr_review > 0:
                    return (True, "Partial auto-approval", approved, hr_review)
                else:
                    return (True, "Full auto-approval", approved, 0)
            else:
                return (False, f"Insufficient leave balance (available: {available_for_type}, needed: {days})", 0, days)
        else:
            approved = min(2, remaining_monthly, available_for_type)
            hr_review = days - approved
            if approved > 0:
                return (True, f"Partial auto-approval ({approved} of {days} days)", approved, hr_review)
            else:
                return (False, "No auto-approval days available", 0, days)


def process_leave_application(leave_data, balance):
    """
    Main function to process leave applications.
    FIXED: Now uses leave month for quota tracking.
    """
    leave_type = leave_data.get('leaveType')
    days = leave_data.get('duration', 0)
    reason = leave_data.get('reason', '')
    from_date_str = leave_data.get('fromDate')
    
    # Get the month of the leave for quota tracking
    monthly_approved, monthly_emergency, leave_month = get_monthly_counters_for_leave(leave_data)
    
    # ========== DETECT IF EMERGENCY ==========
    is_emergency = False
    if from_date_str:
        is_emergency = is_emergency_leave_ai(reason, from_date_str)
    
    # ========== CHECK AUTO-APPROVAL ELIGIBILITY ==========
    can_auto_approve, reason_msg, approved_days, hr_days = check_auto_approval_eligibility(
        leave_data, balance, is_emergency
    )
    
    # ========== CHECK MONTHLY AUTO-APPROVAL LIMIT ==========
    if monthly_approved >= 2 and not is_emergency:
        return handle_monthly_limit_exhausted(leave_data, balance, is_emergency)
    
    if not can_auto_approve:
        if "Monthly auto-approval limit exhausted" in reason_msg:
            return handle_monthly_limit_exhausted(leave_data, balance, is_emergency)
        elif "Insufficient" in reason_msg:
            return suggest_alternative_leaves(leave_data, balance, is_emergency)
        elif "Pattern abuse" in reason_msg:
            pattern_detected, pattern_category, pattern_count = detect_reason_pattern(leave_data)
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
        # Determine which leave type(s) to use
        if leave_type == 'casual':
            casual_available = balance['casual']
            casual_used = min(casual_available, approved_days)
            comp_off_used = approved_days - casual_used
            
            if casual_used > 0:
                balance['casual'] -= casual_used
            if comp_off_used > 0:
                balance['compOff'] -= comp_off_used
            
            used_type = f'{casual_used} Casual Leave'
            if comp_off_used > 0:
                used_type += f' + {comp_off_used} Comp Off'
                
        elif leave_type == 'sick':
            sick_available = balance['sick']
            comp_off_available = balance['compOff']
            casual_available = balance['casual']
            
            sick_used = min(sick_available, approved_days)
            remaining = approved_days - sick_used
            
            comp_off_used = min(comp_off_available, remaining)
            remaining -= comp_off_used
            
            casual_used = min(casual_available, remaining)
            
            if sick_used > 0:
                balance['sick'] -= sick_used
            if comp_off_used > 0:
                balance['compOff'] -= comp_off_used
            if casual_used > 0:
                balance['casual'] -= casual_used
            
            used_parts = []
            if sick_used > 0:
                used_parts.append(f"{sick_used} Sick Leave")
            if comp_off_used > 0:
                used_parts.append(f"{comp_off_used} Comp Off")
            if casual_used > 0:
                used_parts.append(f"{casual_used} Casual Leave")
            
            used_type = " + ".join(used_parts)
            
        elif leave_type == 'compOff':
            balance['compOff'] -= approved_days
            used_type = f'{approved_days} Comp Off'
        
        # FIXED: Update monthly counters for the leave month
        update_monthly_counters(leave_month, approved_days, is_emergency)
        
        # Update the new counters
        new_monthly_approved = session['monthly_auto_approved_by_month'].get(leave_month, 0)
        new_monthly_emergency = session['monthly_emergency_by_month'].get(leave_month, 0)
        
        session['leave_balance'] = balance
        track_applied_leave(leave_data, status='approved')
        session.modified = True
        
        # Build response message
        if hr_days > 0:
            if is_emergency:
                recipient = "HR"
            else:
                recipient = "Manager/HR"
            
            response_msg = f'''✅ {approved_days} day(s) have been auto-approved using {used_type}!
            
The remaining {hr_days} day(s) will require {recipient} approval.

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used for {leave_month}: {new_monthly_approved}/2'''
        else:
            response_msg = f'''✅ Your leave has been auto-approved using {used_type}!

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used for {leave_month}: {new_monthly_approved}/2'''
        
        if is_emergency:
            response_msg += f"\n🚨 Emergency requests used for {leave_month}: {new_monthly_emergency}/1"
        
        return {
            'response': response_msg,
            'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
            'state': 'initial'
        }
    
    # If we reach here, suggest alternatives
    return suggest_alternative_leaves(leave_data, balance, is_emergency)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# HTML Template (unchanged, same as your original)
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

# Initialize session data
def init_session():
    if 'leave_balance' not in session:
        session['leave_balance'] = {
            'casual': 2,
            'sick': 5,
            'compOff': 1
        }
    if 'employee_salary' not in session:
        session['employee_salary'] = 30000
    if 'applied_leaves' not in session:
        session['applied_leaves'] = []
    
    # Performance tracking variables
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
    
    # FIXED: Store monthly counters per month (dictionary keyed by 'YYYY-MM')
    if 'monthly_auto_approved_by_month' not in session:
        session['monthly_auto_approved_by_month'] = {}
    if 'monthly_emergency_by_month' not in session:
        session['monthly_emergency_by_month'] = {}
    
    session.modified = True


def get_monthly_counters_for_leave(leave_data):
    """
    Get the monthly counters for the month of the leave application.
    This allows users to apply for next month's leave even if current month quota is exhausted.
    
    Returns: (monthly_approved: int, monthly_emergency: int, month_key: str)
    """
    from_date_str = leave_data.get('fromDate')
    
    if not from_date_str:
        # Fallback to current month if no FROM date
        current_month = datetime.now().strftime('%Y-%m')
        monthly_approved = session.get('monthly_auto_approved_by_month', {}).get(current_month, 0)
        monthly_emergency = session.get('monthly_emergency_by_month', {}).get(current_month, 0)
        return (monthly_approved, monthly_emergency, current_month)
    
    try:
        from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        # Use the month of the FROM date for quota tracking
        leave_month = from_date.strftime('%Y-%m')
        
        monthly_approved = session.get('monthly_auto_approved_by_month', {}).get(leave_month, 0)
        monthly_emergency = session.get('monthly_emergency_by_month', {}).get(leave_month, 0)
        
        return (monthly_approved, monthly_emergency, leave_month)
    except:
        # Fallback to current month on error
        current_month = datetime.now().strftime('%Y-%m')
        monthly_approved = session.get('monthly_auto_approved_by_month', {}).get(current_month, 0)
        monthly_emergency = session.get('monthly_emergency_by_month', {}).get(current_month, 0)
        return (monthly_approved, monthly_emergency, current_month)


def update_monthly_counters(leave_month, approved_days=0, is_emergency=False):
    """
    Update monthly counters for the specific month.
    
    Args:
        leave_month: Month key in format 'YYYY-MM'
        approved_days: Number of days auto-approved
        is_emergency: Whether this is an emergency request
    """
    if 'monthly_auto_approved_by_month' not in session:
        session['monthly_auto_approved_by_month'] = {}
    if 'monthly_emergency_by_month' not in session:
        session['monthly_emergency_by_month'] = {}
    
    # Update counters for the specific month
    current_approved = session['monthly_auto_approved_by_month'].get(leave_month, 0)
    session['monthly_auto_approved_by_month'][leave_month] = current_approved + approved_days
    
    if is_emergency:
        current_emergency = session['monthly_emergency_by_month'].get(leave_month, 0)
        session['monthly_emergency_by_month'][leave_month] = current_emergency + 1
    
    session.modified = True
    
def check_leave_eligibility():
    """
    Check if employee is eligible for automatic leave approval.
    Returns: (is_eligible: bool, violations: list)
    """
    violations = []
    
    # Rule 1: Last month attendance > 90%
    if session.get('last_month_attendance', 100) <= 90:
        violations.append(f"❌ Last month attendance ({session.get('last_month_attendance')}%) is not above 90%")
    
    # Rule 2: Last month work performance > 90%
    if session.get('last_month_performance', 100) <= 90:
        violations.append(f"❌ Last month work performance ({session.get('last_month_performance')}%) is not above 90%")
    
    # Rule 3: Late coming minutes exceeded (120 minutes limit)
    if session.get('late_minutes_used', 0) >= 120:
        violations.append(f"❌ Late coming limit exceeded ({session.get('late_minutes_used')} minutes used, limit: 120 minutes)")
    
    # Rule 4: Last 3 months average attendance > 85%
    if session.get('last_3_months_attendance', 100) <= 85:
        violations.append(f"❌ Last 3 months average attendance ({session.get('last_3_months_attendance')}%) is not above 85%")
    
    # Rule 5: Pattern recognition - same reason in 3 consecutive months
    if session.get('leave_pattern_violation', False):
        violations.append("❌ Pattern violation: You have applied for leave with similar reasons in the last 3 consecutive months")
    
    is_eligible = len(violations) == 0
    return is_eligible, violations

def is_valid_date_format(date_string):
    """Check if date is in YYYY-MM-DD format"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def is_employee_sick(reason):
    """
    Decide if the employee is sick using the AI model.
    Returns True if classified as 'sick', else False.
    """
    if not reason:
        return False
    label = classify_reason(reason)
    return label == "sick"

def has_date_overlap(new_from_date, new_to_date, applied_leaves):
    """Check if new dates overlap with already applied leaves"""
    try:
        new_from = datetime.strptime(new_from_date, '%Y-%m-%d')
        new_to = datetime.strptime(new_to_date, '%Y-%m-%d')
        
        for applied_leave in applied_leaves:
            if not applied_leave.get('fromDate') or not applied_leave.get('toDate'):
                continue
            
            applied_from = datetime.strptime(applied_leave['fromDate'], '%Y-%m-%d')
            applied_to = datetime.strptime(applied_leave['toDate'], '%Y-%m-%d')
            
            if (new_from <= applied_to) and (new_to >= applied_from):
                return True
        return False
    except (ValueError, TypeError):
        return True

def calculate_days(from_date, to_date):
    """Calculate working days excluding Sundays"""
    try:
        start_date = datetime.strptime(from_date, '%Y-%m-%d')
        end_date = datetime.strptime(to_date, '%Y-%m-%d')
        
        working_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() != 6:
                working_days += 1
            current_date += timedelta(days=1)
        
        return working_days
    except:
        return 0

def calculate_lop_amount(lop_days):
    monthly_salary = session.get('employee_salary', 30000)
    per_day_salary = monthly_salary / 30
    lop_amount = lop_days * per_day_salary
    return round(lop_amount, 2)

def track_applied_leave(leave_data, status='applied'):
    """Track applied leave dates in session."""
    if 'applied_leaves' not in session:
        session['applied_leaves'] = []

    category = categorize_reason_for_pattern(leave_data.get('reason', '') or '')

    session['applied_leaves'].append({
        'fromDate': leave_data.get('fromDate'),
        'toDate': leave_data.get('toDate'),
        'leaveType': leave_data.get('leaveType'),
        'reason': leave_data.get('reason'),
        'category': category,
        'status': status,
        'timestamp': datetime.now().strftime('%Y-%m-%d')
    })
    session.modified = True

# ----------------- HELPER FUNCTIONS FOR LEAVE PROCESSING -----------------

def handle_monthly_limit_exhausted(leave_data, balance, is_emergency=False):
    """Handle case when monthly auto-approval limit is exhausted"""
    leave_type = leave_data.get('leaveType')
    days = leave_data.get('duration', 0)
    reason = leave_data.get('reason', '')
    
    if is_emergency:
        recipient = "HR"
        status = 'pending_hr'
    else:
        recipient = "Manager/HR"
        status = 'pending_manager'
    
    # Check what leaves would be used if approved
    if leave_type == 'casual':
        casual_available = balance['casual']
        comp_off_available = balance['compOff']
        
        if casual_available >= days:
            leave_source = f"{days} Casual Leave"
        elif casual_available + comp_off_available >= days:
            casual_used = min(casual_available, days)
            comp_off_used = days - casual_used
            leave_source = f"{casual_used} Casual Leave + {comp_off_used} Comp Off"
        elif comp_off_available >= days:
            leave_source = f"{days} Comp Off"
        else:
            available = casual_available + comp_off_available
            leave_source = f"{available} days from available balance + {days - available} LOP"
    
    elif leave_type == 'sick':
        sick_available = balance['sick']
        comp_off_available = balance['compOff']
        casual_available = balance['casual']
        total_available = sick_available + comp_off_available + casual_available
        
        if total_available >= days:
            parts = []
            sick_used = min(sick_available, days)
            if sick_used > 0:
                parts.append(f"{sick_used} Sick Leave")
            
            remaining = days - sick_used
            comp_off_used = min(comp_off_available, remaining)
            if comp_off_used > 0:
                parts.append(f"{comp_off_used} Comp Off")
            
            remaining -= comp_off_used
            casual_used = min(casual_available, remaining)
            if casual_used > 0:
                parts.append(f"{casual_used} Casual Leave")
            
            leave_source = " + ".join(parts)
        else:
            leave_source = f"{total_available} days from available balance + {days - total_available} LOP"
    
    elif leave_type == 'compOff':
        if balance['compOff'] >= days:
            leave_source = f"{days} Comp Off"
        else:
            available = balance['compOff']
            leave_source = f"{available} days from available balance + {days - available} LOP"
    
    track_applied_leave(leave_data, status=status)
    
    return {
        'response': f'''📤 Your leave cannot be auto-approved due to monthly limit exhaustion (2/2 days used).

Your application has been sent to {recipient} for approval.

If approved, it will use: {leave_source}

Your current balance (unchanged until approval):
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used: {session.get('monthly_auto_approved_days', 0)}/2''',
        'buttons': [
            {'text': 'Submit for Approval', 'action': 'submitForApproval'},
            {'text': 'Cancel Leave', 'action': 'cancelLeave'}
        ],
        'state': 'waitingDecision',
        'leaveData': leave_data
    }

def suggest_alternative_leaves(leave_data, balance, is_emergency=False):
    """
    Suggest alternative leave options when primary type has insufficient balance.
    """
    leave_type = leave_data.get('leaveType')
    days = leave_data.get('duration', 0)
    
    # Check monthly auto-approval limit first
    monthly_approved = session.get('monthly_auto_approved_days', 0)
    if monthly_approved >= 2 and not is_emergency:
        return handle_monthly_limit_exhausted(leave_data, balance, is_emergency)
    
    # Calculate total available balance for this leave type
    if leave_type == 'casual':
        total_available = balance['casual'] + balance['compOff']
    elif leave_type == 'sick':
        total_available = balance['sick'] + balance['compOff'] + balance['casual']
    elif leave_type == 'compOff':
        total_available = balance['compOff']
    else:
        total_available = 0
    
    available_in_current = total_available
    
    if available_in_current > 0:
        shortfall = max(0, days - available_in_current)
        
        # If sick leave, show all available options
        if leave_type == 'sick':
            sick_available = balance['sick']
            comp_off_available = balance['compOff']
            casual_available = balance['casual']
            
            if sick_available >= days:
                return {
                    'response': f'''You have {sick_available} day(s) of Sick Leave available.
This will use: {days} Sick Leave''',
                    'buttons': [
                        {'text': f'Yes, Use {days} Sick Leave', 'action': 'useSickOnly'},
                        {'text': 'No, Proceed with Manager/HR Review', 'action': 'submitForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision'
                }
            elif sick_available > 0 and (comp_off_available + casual_available) >= (days - sick_available):
                return {
                    'response': f'''You have limited Sick Leave balance ({sick_available} days).
Would you like to use other available balances?

Available balances (in priority order):
- Sick Leave: {sick_available} days
- Comp Off: {comp_off_available} days
- Casual Leave: {casual_available} days

This will use: {sick_available} Sick Leave + {days - sick_available} from Comp Off/Casual Leave''',
                    'buttons': [
                        {'text': f'Yes, Use All Available Balances', 'action': 'useCombinationForSick'},
                        {'text': 'No, Proceed with Manager/HR Review', 'action': 'submitForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision'
                }
            elif (comp_off_available + casual_available) >= days:
                return {
                    'response': f'''You don't have Sick Leave balance, but you can use other balances:

Available balances (in priority order):
- Comp Off: {comp_off_available} days
- Casual Leave: {casual_available} days

This will use: Comp Off + Casual Leave (as needed for sick leave)''',
                    'buttons': [
                        {'text': f'Yes, Use Comp Off/Casual Leave', 'action': 'useCasualCompOffForSick'},
                        {'text': 'No, Proceed with Manager/HR Review', 'action': 'submitForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision'
                }
        
        elif leave_type == 'casual':
            casual_available = balance['casual']
            comp_off_available = balance['compOff']
            
            if casual_available >= days:
                return {
                    'response': f'''You have {casual_available} day(s) of Casual Leave available.
This will use: {days} Casual Leave''',
                    'buttons': [
                        {'text': f'Yes, Use {days} Casual Leave', 'action': 'useCasualOnly'},
                        {'text': 'No, Proceed with Manager/HR Review', 'action': 'submitForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision'
                }
            elif casual_available > 0 and comp_off_available >= shortfall:
                return {
                    'response': f'''You have {casual_available} day(s) of Casual Leave available.
Would you like to use Comp Off for the remaining {shortfall} day(s)?

This will use: {casual_available} Casual Leave + {shortfall} Comp Off''',
                    'buttons': [
                        {'text': f'Yes, Use {casual_available} Casual + {shortfall} Comp Off', 'action': 'usePartialCasualWithCompOff'},
                        {'text': 'No, Proceed with Manager/HR Review', 'action': 'submitForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision'
                }
            elif comp_off_available >= days:
                return {
                    'response': f'''You don't have Casual Leave, but you can use {comp_off_available} Comp Off.

This will use: {days} Comp Off''',
                    'buttons': [
                        {'text': f'Yes, Use {days} Comp Off', 'action': 'useCompOffOnly'},
                        {'text': 'No, Proceed with Manager/HR Review', 'action': 'submitForApproval'},
                        {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                    ],
                    'state': 'waitingDecision'
                }
    
    # Check total available balance
    total_available = available_in_current
        
    if total_available > 0:
        return check_partial_lop_options(leave_data, balance, is_emergency)
    else:
        # NO balance at all
        if is_emergency and days == 1:
            return {
                'response': '''🚨 EMERGENCY DETECTED: You have no leave balance, but emergency override allows 1 day auto-approval.

Your 1 day emergency leave has been auto-approved (LOP will apply).''',
                'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
                'state': 'initial'
            }
        else:
            return {
                'response': f'''⚠️ You don't have sufficient leave balance for any leave type.

Casual Leave: {balance['casual']} days
Sick Leave: {balance['sick']} days
Comp Off: {balance['compOff']} days

This will require HR approval and result in Loss of Pay (LOP).''',
                'buttons': [
                    {'text': 'Submit for HR Approval', 'action': 'submitForApproval'},
                    {'text': 'Cancel Leave', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision'
            }

def check_partial_lop_options(leave_data, balance, is_emergency=False):
    """Check if partial LOP is possible with available balances"""
    days = leave_data.get('duration', 0)
    leave_type = leave_data.get('leaveType')
    
    # Calculate total available days from all applicable leave types
    if leave_type == 'casual':
        total_available = balance['casual'] + balance['compOff']
    elif leave_type == 'sick':
        total_available = balance['sick'] + balance['compOff'] + balance['casual']
    elif leave_type == 'compOff':
        total_available = balance['compOff']
    else:
        total_available = 0
    
    if total_available > 0:
        lop_days = max(0, days - total_available)
        
        # Build available types list
        available_types = []
        if leave_type == 'casual':
            if balance['casual'] > 0:
                available_types.append(('Casual Leave', balance['casual']))
            if balance['compOff'] > 0:
                available_types.append(('Comp Off', balance['compOff']))
        elif leave_type == 'sick':
            if balance['sick'] > 0:
                available_types.append(('Sick Leave', balance['sick']))
            if balance['compOff'] > 0:
                available_types.append(('Comp Off', balance['compOff']))
            if balance['casual'] > 0:
                available_types.append(('Casual Leave', balance['casual']))
        elif leave_type == 'compOff' and balance['compOff'] > 0:
            available_types.append(('Comp Off', balance['compOff']))
        
        balance_text = "\n".join([f"- {leave_type}: {avail_days} day(s)" for leave_type, avail_days in available_types])
        
        if lop_days > 0:
            lop_amount = calculate_lop_amount(lop_days)
            response_text = f'''⚠️ You have some leave balance available, but partial LOP is required:

{balance_text}

This will use all available balance + {lop_days} LOP day(s) (₹{lop_amount})
Requires HR approval (LOP is never auto-approved).

Would you like to proceed?'''
            
            return {
                'response': response_text,
                'buttons': [
                    {'text': 'Yes, Submit for HR Approval', 'action': 'submitForApproval'},
                    {'text': 'No, Cancel', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision'
            }
        else:
            response_text = f'''You have sufficient leave balance available:

{balance_text}

This will use all available balance to cover {days} day(s).

Would you like to proceed?'''
            
            return {
                'response': response_text,
                'buttons': [
                    {'text': 'Yes, Submit for Approval', 'action': 'submitForApproval'},
                    {'text': 'No, Cancel', 'action': 'cancelLeave'}
                ],
                'state': 'waitingDecision'
            }
    
    # No balances available, proceed to full LOP
    return proceed_to_lop(leave_data, balance, is_emergency)

def proceed_to_lop(leave_data, balance, is_emergency=False):
    """Handle full LOP case - NO balance at all"""
    days = leave_data.get('duration', 0)
    lop_amount = calculate_lop_amount(days)
    
    if is_emergency and days == 1:
        # Emergency override: 1 day auto-approved even with no balance
        session['monthly_emergency_requests'] += 1
        session['monthly_auto_approved_days'] += 1
        track_applied_leave(leave_data, status='approved')
        session.modified = True
        
        return {
            'response': f'''🚨 EMERGENCY OVERRIDE: Your 1 day emergency leave has been auto-approved!

📊 Monthly auto-approved days used: {session['monthly_auto_approved_days']}/2
🚨 Emergency requests used: {session['monthly_emergency_requests']}/1

⚠️ Note: This will result in Loss of Pay (₹{lop_amount}) as you have no leave balance.''',
            'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
            'state': 'initial'
        }
    
    if is_emergency:
        recipient = "HR"
    else:
        recipient = "Manager/HR"
    
    return {
        'response': f'''⚠️ You don't have sufficient leave balance for any leave type.

Current balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

This will affect your KPI and ₹{lop_amount} LOP will be deducted from your salary.
This request requires {recipient} approval (LOP is never auto-approved).

Do you want to proceed to {recipient} review?''',
        'buttons': [
            {'text': f'Yes, Submit for {recipient} Approval', 'action': 'submitForApproval'},
            {'text': 'No, Cancel', 'action': 'cancelLeave'}
        ],
        'state': 'waitingDecision'
    }

def process_partial_combination(leave_data, balance, primary_type, secondary_type):
    """Process leave using combination of two leave types"""
    days = leave_data.get('duration', 0)
    is_sick_reason = is_employee_sick(leave_data.get('reason', ''))
    
    # Calculate how many days can be used from each type
    primary_available = balance[primary_type]
    primary_used = min(primary_available, days)
    remaining = days - primary_used
    
    secondary_used = min(balance[secondary_type], remaining)
    remaining_after_secondary = remaining - secondary_used
    
    # Check eligibility for auto-approval
    is_eligible, violations = check_leave_eligibility()
    if not is_eligible:
        violation_text = "\n".join(violations)
        
        # Still deduct the balance but require manager approval
        balance[primary_type] -= primary_used
        balance[secondary_type] -= secondary_used
        session['leave_balance'] = balance
        track_applied_leave(leave_data, status='pending_manager')
        
        # Build response for Manager/HR approval
        primary_display = 'Comp Off' if primary_type == 'compOff' else primary_type.title() + ' Leave'
        secondary_display = 'Comp Off' if secondary_type == 'compOff' else secondary_type.title() + ' Leave'
        
        breakdown = f"- {primary_used} day(s) {primary_display}\n- {secondary_used} day(s) {secondary_display}"
        
        if remaining_after_secondary > 0:
            lop_amount = calculate_lop_amount(remaining_after_secondary)
            breakdown += f"\n- LOP: {remaining_after_secondary} day(s) (₹{lop_amount})"
        
        response_msg = f'''⚠️ Your leave cannot be auto-approved due to violations:

{violation_text}

Your leave has been submitted for manager approval.

Leave breakdown:
{breakdown}

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days'''
        
        return {
            'response': response_msg,
            'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
            'state': 'initial'
        }
    
    # Check monthly auto-approval limits
    monthly_approved = session.get('monthly_auto_approved_days', 0)
    remaining_monthly = 2 - monthly_approved
    
    # RULE 1-3: Monthly auto-approval limit
    if remaining_monthly <= 0:
        # Monthly limit exhausted, need Manager/HR approval
        balance[primary_type] -= primary_used
        balance[secondary_type] -= secondary_used
        session['leave_balance'] = balance
        
        track_applied_leave(leave_data, status='pending_manager')
        
        # Build response for Manager/HR approval
        primary_display = 'Comp Off' if primary_type == 'compOff' else primary_type.title() + ' Leave'
        secondary_display = 'Comp Off' if secondary_type == 'compOff' else secondary_type.title() + ' Leave'
        
        breakdown = f"- {primary_used} day(s) {primary_display}\n- {secondary_used} day(s) {secondary_display}"
        
        if remaining_after_secondary > 0:
            lop_amount = calculate_lop_amount(remaining_after_secondary)
            breakdown += f"\n- LOP: {remaining_after_secondary} day(s) (₹{lop_amount})"
        
        response_msg = f'''📤 Your leave application has been submitted and is waiting for Manager/HR approval.

Reason: Monthly auto-approval limit exhausted (2/2 days used)

Leave breakdown:
{breakdown}

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days'''
        
        return {
            'response': response_msg,
            'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
            'state': 'initial'
        }
    
    # Calculate how many days can be auto-approved
    primary_available_for_auto = min(primary_used, remaining_monthly)
    remaining_monthly -= primary_available_for_auto
    secondary_available_for_auto = min(secondary_used, remaining_monthly)
    
    total_auto_approved = primary_available_for_auto + secondary_available_for_auto
    
    if remaining_after_secondary > 0:
        # Has LOP days - cannot auto-approve
        balance[primary_type] -= primary_used
        balance[secondary_type] -= secondary_used
        session['leave_balance'] = balance
        
        track_applied_leave(leave_data, status='pending_manager')
        
        lop_amount = calculate_lop_amount(remaining_after_secondary)
        
        primary_display = 'Comp Off' if primary_type == 'compOff' else primary_type.title() + ' Leave'
        secondary_display = 'Comp Off' if secondary_type == 'compOff' else secondary_type.title() + ' Leave'
        
        breakdown = f"- {primary_used} day(s) {primary_display}\n- {secondary_used} day(s) {secondary_display}"
        breakdown += f"\n- LOP: {remaining_after_secondary} day(s) (₹{lop_amount})"
        
        response_msg = f'''⚠️ Your leave cannot be fully auto-approved due to LOP requirements.

The {primary_used + secondary_used} day(s) using available balance have been approved.
The remaining {remaining_after_secondary} LOP day(s) will require Manager/HR approval.

Leave breakdown:
{breakdown}

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days'''
        
        return {
            'response': response_msg,
            'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
            'state': 'initial'
        }
    
    # All days can be covered by available balance
    # Update monthly counter
    session['monthly_auto_approved_days'] += total_auto_approved
    
    # Deduct leaves
    balance[primary_type] -= primary_used
    balance[secondary_type] -= secondary_used
    session['leave_balance'] = balance
    track_applied_leave(leave_data, status='approved')
    
    # Build response
    primary_display = 'Comp Off' if primary_type == 'compOff' else primary_type.title() + ' Leave'
    secondary_display = 'Comp Off' if secondary_type == 'compOff' else secondary_type.title() + ' Leave'
    
    breakdown = f"- {primary_used} day(s) {primary_display}\n- {secondary_used} day(s) {secondary_display}"
    
    response_msg = f'''✅ Your leave has been auto-approved!

Leave breakdown:
{breakdown}

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used: {session['monthly_auto_approved_days']}/2'''
    
    return {
        'response': response_msg,
        'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
        'state': 'initial'
    }

def process_sick_leave_with_combination(leave_data, balance):
    """Process sick leave using combination of sick + other balances"""
    days = leave_data.get('duration', 0)
    
    # Calculate how many days from each balance (priority: sick -> comp off -> casual)
    sick_available = balance['sick']
    comp_off_available = balance['compOff']
    casual_available = balance['casual']
    
    sick_used = min(sick_available, days)
    remaining = days - sick_used
    
    comp_off_used = min(comp_off_available, remaining)
    remaining -= comp_off_used
    
    casual_used = min(casual_available, remaining)
    remaining_after_all = remaining - casual_used
    
    # Check monthly auto-approval limits
    monthly_approved = session.get('monthly_auto_approved_days', 0)
    remaining_monthly = 2 - monthly_approved
    
    if remaining_monthly <= 0:
        # Monthly limit exhausted
        return handle_monthly_limit_exhausted(leave_data, balance, False)
    
    # Calculate how many days can be auto-approved
    total_auto_approved = min(days - remaining_after_all, remaining_monthly)
    
    if remaining_after_all > 0:
        # Has LOP days - cannot auto-approve
        return check_partial_lop_options(leave_data, balance, False)
    
    # All days can be covered by available balance
    # Update monthly counter
    session['monthly_auto_approved_days'] += total_auto_approved
    
    # Deduct leaves
    balance['sick'] -= sick_used
    balance['compOff'] -= comp_off_used
    balance['casual'] -= casual_used
    session['leave_balance'] = balance
    track_applied_leave(leave_data, status='approved')
    session.modified = True
    
    # Build response
    used_parts = []
    if sick_used > 0:
        used_parts.append(f"{sick_used} Sick Leave")
    if comp_off_used > 0:
        used_parts.append(f"{comp_off_used} Comp Off")
    if casual_used > 0:
        used_parts.append(f"{casual_used} Casual Leave")
    
    breakdown = "\n".join([f"- {part}" for part in used_parts])
    
    response_msg = f'''✅ Your leave has been auto-approved!

Leave breakdown:
{breakdown}

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used: {session['monthly_auto_approved_days']}/2'''
    
    return {
        'response': response_msg,
        'buttons': [{'text': 'Start Again', 'action': 'startAgain'}],
        'state': 'initial'
    }

# ----------------- FLASK ROUTES -----------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/init')
def init():
    init_session()
    return jsonify({
        'response': 'Welcome! 👋 How can I help you today?',
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
    balance = session['leave_balance']
    
    response = {'needsInput': False, 'buttons': None, 'state': 'initial', 'leaveData': leave_data}
    
    if action == 'applyLeave':
        response['response'] = 'Please select the type of leave you want to apply:'
        response['buttons'] = [
            {'text': 'Casual Leave or Sick Leave', 'action': 'casualOrSick'},
            {'text': 'Comp Off Leave', 'action': 'compOff'}
        ]
        response['state'] = 'selectLeaveType'
        
    elif action == 'checkBalance':
        response['response'] = f'''Your current leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days'''
        response['buttons'] = [
            {'text': 'Apply for a Leave', 'action': 'applyLeave'},
            {'text': 'Check Leave Balance', 'action': 'checkBalance'}
        ]
        
    elif action == 'canApply':
        has_leaves = balance['casual'] > 0 or balance['sick'] > 0 or balance['compOff'] > 0
        if has_leaves:
            response['response'] = f'''Yes, you can apply for leave! You have:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days'''
        else:
            response['response'] = 'You currently have no leave balance. Applying for leave will result in Loss of Pay (LOP) and may affect your KPI.'
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
        
        # FIXED: Show current and next month's counters
        current_month = datetime.now().strftime('%Y-%m')
        next_month = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).strftime('%Y-%m')
        
        current_month_approved = session.get('monthly_auto_approved_by_month', {}).get(current_month, 0)
        current_month_emergency = session.get('monthly_emergency_by_month', {}).get(current_month, 0)
        
        next_month_approved = session.get('monthly_auto_approved_by_month', {}).get(next_month, 0)
        next_month_emergency = session.get('monthly_emergency_by_month', {}).get(next_month, 0)
        
        current_month_name = datetime.now().strftime('%B %Y')
        next_month_name = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).strftime('%B %Y')
        
        metrics = f'''📊 Your Performance Metrics:

    📅 Last Month Attendance: {session.get('last_month_attendance', 0)}%
    💼 Last Month Performance: {session.get('last_month_performance', 0)}%
    ⏰ Late Minutes Used: {session.get('late_minutes_used', 0)}/120 minutes
    📈 Last 3 Months Avg Attendance: {session.get('last_3_months_attendance', 0)}%
    🔍 Pattern Violation: {'Yes' if session.get('leave_pattern_violation', False) else 'No'}

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
        result = process_leave_application(leave_data, balance)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'switchToSick':
        leave_data['leaveType'] = 'sick'
        result = process_leave_application(leave_data, balance)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'switchToCompoff':
        leave_data['leaveType'] = 'compOff'
        result = process_leave_application(leave_data, balance)
        response.update(result)
        response['leaveData'] = leave_data
        
    elif action == 'useSickOnly':
        # New action: Use sick leave only
        leave_type = leave_data.get('leaveType')
        days = leave_data.get('duration', 0)
        
        if leave_type == 'sick' and balance['sick'] >= days:
            # Check monthly auto-approval limit
            monthly_approved = session.get('monthly_auto_approved_days', 0)
            if monthly_approved >= 2:
                return handle_monthly_limit_exhausted(leave_data, balance, False)
            
            # Auto-approve if eligible
            balance['sick'] -= days
            session['monthly_auto_approved_days'] += days
            session['leave_balance'] = balance
            track_applied_leave(leave_data, status='approved')
            session.modified = True
            
            response['response'] = f'''✅ Your leave has been auto-approved using {days} Sick Leave!

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used: {session['monthly_auto_approved_days']}/2'''
            response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
            response['state'] = 'initial'
        else:
            response['response'] = 'Unable to process request. Please try again.'
            response['buttons'] = [
                {'text': 'Apply for a Leave', 'action': 'applyLeave'},
                {'text': 'Check Leave Balance', 'action': 'checkBalance'}
            ]
            response['state'] = 'initial'
        
    elif action == 'useCasualOnly':
        # New action: Use casual leave only
        leave_type = leave_data.get('leaveType')
        days = leave_data.get('duration', 0)
        
        if leave_type == 'casual' and balance['casual'] >= days:
            # Check monthly auto-approval limit
            monthly_approved = session.get('monthly_auto_approved_days', 0)
            if monthly_approved >= 2:
                return handle_monthly_limit_exhausted(leave_data, balance, False)
            
            # Auto-approve if eligible
            balance['casual'] -= days
            session['monthly_auto_approved_days'] += days
            session['leave_balance'] = balance
            track_applied_leave(leave_data, status='approved')
            session.modified = True
            
            response['response'] = f'''✅ Your leave has been auto-approved using {days} Casual Leave!

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used: {session['monthly_auto_approved_days']}/2'''
            response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
            response['state'] = 'initial'
        else:
            response['response'] = 'Unable to process request. Please try again.'
            response['buttons'] = [
                {'text': 'Apply for a Leave', 'action': 'applyLeave'},
                {'text': 'Check Leave Balance', 'action': 'checkBalance'}
            ]
            response['state'] = 'initial'
    
    elif action == 'useCompOffOnly':
        # New action: Use comp off only
        leave_type = leave_data.get('leaveType')
        days = leave_data.get('duration', 0)
        
        if leave_type in ['casual', 'compOff'] and balance['compOff'] >= days:
            # Check monthly auto-approval limit
            monthly_approved = session.get('monthly_auto_approved_days', 0)
            if monthly_approved >= 2:
                return handle_monthly_limit_exhausted(leave_data, balance, False)
            
            # Auto-approve if eligible
            balance['compOff'] -= days
            session['monthly_auto_approved_days'] += days
            session['leave_balance'] = balance
            track_applied_leave(leave_data, status='approved')
            session.modified = True
            
            response['response'] = f'''✅ Your leave has been auto-approved using {days} Comp Off!

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used: {session['monthly_auto_approved_days']}/2'''
            response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
            response['state'] = 'initial'
        else:
            response['response'] = 'Unable to process request. Please try again.'
            response['buttons'] = [
                {'text': 'Apply for a Leave', 'action': 'applyLeave'},
                {'text': 'Check Leave Balance', 'action': 'checkBalance'}
            ]
            response['state'] = 'initial'
        
    elif action == 'usePartialCasualWithCompOff':
        result = process_partial_combination(leave_data, balance, primary_type='casual', secondary_type='compOff')
        response.update(result)
    
    elif action == 'useCombinationForSick':
        result = process_sick_leave_with_combination(leave_data, balance)
        response.update(result)
    
    elif action == 'useCasualCompOffForSick':
        # For sick leave without sick balance, use comp off and casual (priority: comp off first)
        leave_type = leave_data.get('leaveType')
        days = leave_data.get('duration', 0)
        
        # Use comp off first, then casual
        comp_off_available = balance['compOff']
        casual_available = balance['casual']
        
        comp_off_used = min(comp_off_available, days)
        remaining = days - comp_off_used
        casual_used = min(casual_available, remaining)
        
        # Check monthly auto-approval limit
        monthly_approved = session.get('monthly_auto_approved_days', 0)
        remaining_monthly = 2 - monthly_approved
        
        if remaining_monthly <= 0:
            return handle_monthly_limit_exhausted(leave_data, balance, False)
        
        # Update monthly counter
        total_used = comp_off_used + casual_used
        session['monthly_auto_approved_days'] += total_used
        
        # Deduct balances
        balance['compOff'] -= comp_off_used
        balance['casual'] -= casual_used
        session['leave_balance'] = balance
        track_applied_leave(leave_data, status='approved')
        session.modified = True
        
        # Build response
        used_parts = []
        if comp_off_used > 0:
            used_parts.append(f"{comp_off_used} Comp Off")
        if casual_used > 0:
            used_parts.append(f"{casual_used} Casual Leave")
        
        used_type = " + ".join(used_parts)
        
        response['response'] = f'''✅ Your leave has been auto-approved using {used_type}!

Your updated leave balance:
📅 Casual Leave: {balance['casual']} days
🏥 Sick Leave: {balance['sick']} days
⏰ Comp Off: {balance['compOff']} days

📊 Monthly auto-approved days used: {session['monthly_auto_approved_days']}/2'''
        response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
        response['state'] = 'initial'
        
    elif action == 'submitForApproval':
        # Submit for HR/Manager approval
        leave_type = leave_data.get('leaveType')
        days = leave_data.get('duration', 0)
        reason = leave_data.get('reason', '')
        from_date_str = leave_data.get('fromDate')
        
        # Check if emergency (considering date now)
        is_emergency = False
        if from_date_str:
            is_emergency = is_emergency_leave_ai(reason, from_date_str)
        
        if is_emergency:
            recipient = "HR"
            status = 'pending_hr'
        else:
            recipient = "Manager/HR"
            status = 'pending_manager'
        
        # Check what leaves would be used if approved - FIXED: Use appropriate balances
        if leave_type == 'casual':
            # Use casual leave first, then comp off
            casual_available = balance['casual']
            comp_off_available = balance['compOff']
            
            if casual_available >= days:
                leave_source = f"{days} Casual Leave"
            elif casual_available + comp_off_available >= days:
                casual_used = min(casual_available, days)
                comp_off_used = days - casual_used
                leave_source = f"{casual_used} Casual Leave + {comp_off_used} Comp Off"
            elif comp_off_available >= days:
                leave_source = f"{days} Comp Off"
            else:
                available = casual_available + comp_off_available
                leave_source = f"{available} days from available balance + {days - available} LOP"
        
        elif leave_type == 'sick':
            # FIXED: Sick leave can use ALL THREE balances (priority: sick -> comp off -> casual)
            sick_available = balance['sick']
            comp_off_available = balance['compOff']
            casual_available = balance['casual']
            total_available = sick_available + comp_off_available + casual_available
            
            if total_available >= days:
                # Build breakdown (priority order)
                parts = []
                sick_used = min(sick_available, days)
                if sick_used > 0:
                    parts.append(f"{sick_used} Sick Leave")
                
                remaining = days - sick_used
                comp_off_used = min(comp_off_available, remaining)
                if comp_off_used > 0:
                    parts.append(f"{comp_off_used} Comp Off")
                
                remaining -= comp_off_used
                casual_used = min(casual_available, remaining)
                if casual_used > 0:
                    parts.append(f"{casual_used} Casual Leave")
                
                leave_source = " + ".join(parts)
            else:
                leave_source = f"{total_available} days from available balance + {days - total_available} LOP"
        
        elif leave_type == 'compOff':
            if balance['compOff'] >= days:
                leave_source = f"{days} Comp Off"
            else:
                available = balance['compOff']
                leave_source = f"{available} days from available balance + {days - available} LOP"
        
        track_applied_leave(leave_data, status=status)
        
        response['response'] = f'''📤 Your leave application has been submitted and is waiting for {recipient} approval.

Application Details:
- Leave Type: {'Comp Off' if leave_type == 'compOff' else leave_type.title() + ' Leave'}
- Duration: {days} day(s)
- Reason: {reason}

If approved, it will use: {leave_source}

Your leave balance remains unchanged pending approval.'''
        
        response['buttons'] = [{'text': 'Start Again', 'action': 'startAgain'}]
        response['state'] = 'initial'
        
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
        response['response'] = 'Welcome back! 👋 How can I help you today?'
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
    balance = session['leave_balance']
    applied_leaves = session['applied_leaves']
    
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
        is_sick_reason = is_employee_sick(reason)
        
        # CRITICAL FIX: Check emergency ONLY after we have the FROM date
        is_emergency = is_emergency_leave_ai(reason, message)
        
        if leave_type == 'sick' or is_sick_reason:
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
                    response['response'] = '❌ For casual leave, FROM date must be TOMORROW or later (one day prior rule). Please enter a valid FROM date:'
                response['needsInput'] = True
                response['state'] = 'enterFromDate'
                return jsonify(response)
        
        elif leave_type == 'compOff':
            if not is_valid_casual_comp_off_from_date(message, is_emergency):
                if is_emergency:
                    response['response'] = "❌ For EMERGENCY comp off, FROM date MUST be TODAY. Please enter today's date:"
                else:
                    response['response'] = '❌ For Comp Off leaves, FROM date must be today or a future date. Please enter a valid FROM date:'
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
        
        if has_date_overlap(leave_data.get('fromDate'), message, applied_leaves):
            response['response'] = '❌ These dates overlap with your already applied leaves! Please choose different dates:'
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
        
        if days < calendar_days:
            if is_past_leave:
                response['response'] = f"You are applying for {days} working day(s) (excluding Sundays) out of {calendar_days} calendar day(s) from {leave_data['fromDate']} to {message}.\n\n⚠️ Note: This is a past leave application."
            else:
                response['response'] = f"You are applying for {days} working day(s) (excluding Sundays) out of {calendar_days} calendar day(s) from {leave_data['fromDate']} to {message}."
        else:
            if is_past_leave:
                response['response'] = f"You are applying for {days} day(s) of leave from {leave_data['fromDate']} to {message}.\n\n⚠️ Note: This is a past leave application."
            else:
                response['response'] = f"You are applying for {days} day(s) of leave from {leave_data['fromDate']} to {message}."
        
        result = process_leave_application(leave_data, balance)
        response.update(result)
        response['leaveData'] = leave_data
        
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=8083)
