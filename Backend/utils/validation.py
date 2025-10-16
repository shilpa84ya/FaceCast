"""
Input Validation and Sanitization Module for FaceCast
Provides comprehensive validation for all user inputs
"""

import re
import html
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any
import bleach

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class InputValidator:
    """Comprehensive input validation class"""
    
    # Regex patterns for validation
    PATTERNS = {
        'voter_id': r'^[A-Z]{3}[0-9]{7}$',
        'aadhaar': r'^[0-9]{12}$',
        'phone': r'^[6-9][0-9]{9}$',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'name': r'^[a-zA-Z\s\.\-\']{2,100}$',
        'address': r'^[a-zA-Z0-9\s\.\,\-\#\/]{5,200}$',
        'password': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    }
    
    # Allowed HTML tags for sanitization
    ALLOWED_TAGS = []
    ALLOWED_ATTRIBUTES = {}
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input to prevent XSS attacks"""
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")
        
        # Remove HTML tags and escape special characters
        sanitized = bleach.clean(value, tags=InputValidator.ALLOWED_TAGS, 
                               attributes=InputValidator.ALLOWED_ATTRIBUTES)
        sanitized = html.escape(sanitized)
        
        # Trim whitespace and limit length
        sanitized = sanitized.strip()[:max_length]
        
        return sanitized
    
    @staticmethod
    def validate_voter_id(voter_id: str) -> str:
        """Validate voter ID format (3 letters + 7 digits)"""
        if not voter_id:
            raise ValidationError("Voter ID is required")
        
        voter_id = InputValidator.sanitize_string(voter_id, 10).upper()
        
        if not re.match(InputValidator.PATTERNS['voter_id'], voter_id):
            raise ValidationError("Invalid Voter ID format. Expected 3 letters and 7 digits (e.g., ABC1234567)")
        
        return voter_id
    
    @staticmethod
    def validate_aadhaar(aadhaar: str) -> str:
        """Validate Aadhaar number format (12 digits)"""
        if not aadhaar:
            raise ValidationError("Aadhaar number is required")
        
        aadhaar = InputValidator.sanitize_string(aadhaar, 12)
        aadhaar = re.sub(r'\D', '', aadhaar)  # Remove non-digits
        
        if not re.match(InputValidator.PATTERNS['aadhaar'], aadhaar):
            raise ValidationError("Invalid Aadhaar number format. Expected 12 digits")
        
        return aadhaar
    
    @staticmethod
    def validate_phone(phone: str, required: bool = False) -> Optional[str]:
        """Validate Indian phone number format"""
        if not phone:
            if required:
                raise ValidationError("Phone number is required")
            return None
        
        phone = InputValidator.sanitize_string(phone, 10)
        phone = re.sub(r'\D', '', phone)  # Remove non-digits
        
        if not re.match(InputValidator.PATTERNS['phone'], phone):
            raise ValidationError("Invalid phone number format. Expected a 10-digit Indian number starting with 6-9")
        
        return phone
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address format"""
        if not email:
            raise ValidationError("Email address is required")
        
        email = InputValidator.sanitize_string(email, 254).lower()
        
        if not re.match(InputValidator.PATTERNS['email'], email):
            raise ValidationError("Invalid email address format")
        
        return email
    
    @staticmethod
    def validate_name(name: str) -> str:
        """Validate person name"""
        if not name:
            raise ValidationError("Name is required")
        
        name = InputValidator.sanitize_string(name, 100)
        
        if not re.match(InputValidator.PATTERNS['name'], name):
            raise ValidationError("Invalid name format. Only letters, spaces, dots, hyphens, and apostrophes allowed")
        
        if len(name) < 2:
            raise ValidationError("Name must be at least 2 characters long")
        
        return name.title()  # Capitalize properly
    
    @staticmethod
    def validate_address(address: str) -> str:
        """Validate address"""
        if not address:
            raise ValidationError("Address is required")
        
        address = InputValidator.sanitize_string(address, 200)
        
        if not re.match(InputValidator.PATTERNS['address'], address):
            raise ValidationError("Invalid address format")
        
        if len(address) < 5:
            raise ValidationError("Address must be at least 5 characters long")
        
        return address
    
    @staticmethod
    def validate_gender(gender: str) -> str:
        """Validate gender selection"""
        if not gender:
            raise ValidationError("Gender is required")
        
        gender = InputValidator.sanitize_string(gender, 10)
        valid_genders = ['Male', 'Female', 'Other']
        
        if gender not in valid_genders:
            raise ValidationError(f"Invalid gender. Must be one of: {', '.join(valid_genders)}")
        
        return gender
    
    @staticmethod
    def validate_date_of_birth(dob_string: str) -> Tuple[str, int]:
        """Validate date of birth and calculate age"""
        if not dob_string:
            raise ValidationError("Date of birth is required")
        
        try:
            dob = datetime.strptime(dob_string, '%Y-%m-%d').date()
        except ValueError:
            raise ValidationError("Invalid date format. Use YYYY-MM-DD")
        
        # Check if date is not in the future
        if dob > date.today():
            raise ValidationError("Date of birth cannot be in the future")
        
        # Calculate age
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        
        # Check minimum age requirement
        if age < 18:
            raise ValidationError("You must be 18 years or older to register")
        
        # Check maximum reasonable age
        if age > 120:
            raise ValidationError("Invalid date of birth")
        
        return dob_string, age
    
    @staticmethod
    def validate_password(password: str) -> str:
        """Validate password strength"""
        if not password:
            raise ValidationError("Password is required")
        
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long")
        
        if not re.match(InputValidator.PATTERNS['password'], password):
            raise ValidationError("Password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character")
        
        return password
    
    @staticmethod
    def validate_image_data_url(image_data_url: str) -> str:
        """Validate base64 image data URL"""
        if not image_data_url:
            raise ValidationError("Image data is required")
        
        # Check if it's a valid data URL
        if not image_data_url.startswith('data:image/'):
            raise ValidationError("Invalid image format")
        
        # Check if it contains base64 data
        if 'base64,' not in image_data_url:
            raise ValidationError("Invalid image encoding")
        
        # Check size (limit to 5MB)
        if len(image_data_url) > 5 * 1024 * 1024:
            raise ValidationError("Image size too large. Maximum 5MB allowed")
        
        return image_data_url
    
    @staticmethod
    def validate_election_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate election creation data"""
        validated = {}
        
        # Title validation
        if not data.get('title'):
            raise ValidationError("Election title is required")
        validated['title'] = InputValidator.sanitize_string(data['title'], 200)
        
        # Description validation
        validated['description'] = InputValidator.sanitize_string(data.get('description', ''), 1000)
        
        # Date validation
        if not data.get('start_date'):
            raise ValidationError("Start date is required")
        if not data.get('end_date'):
            raise ValidationError("End date is required")
        
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        except ValueError:
            raise ValidationError("Invalid date format. Use YYYY-MM-DD")
        
        if end_date <= start_date:
            raise ValidationError("End date must be after start date")
        
        validated['start_date'] = data['start_date']
        validated['end_date'] = data['end_date']
        validated['start_time'] = data.get('start_time', '00:00')
        
        # Demo election validation
        validated['is_demo'] = bool(data.get('is_demo', False))
        validated['demo_duration_minutes'] = int(data.get('demo_duration_minutes', 2))
        validated['use_quick_duration'] = bool(data.get('use_quick_duration', False))
        
        return validated
    
    @staticmethod
    def validate_candidate_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate candidate data"""
        validated = {}
        
        # Name validation
        if not data.get('name'):
            raise ValidationError("Candidate name is required")
        validated['name'] = InputValidator.validate_name(data['name'])
        
        # Election ID validation
        if not data.get('election_id'):
            raise ValidationError("Election ID is required")
        try:
            validated['election_id'] = int(data['election_id'])
        except (ValueError, TypeError):
            raise ValidationError("Invalid election ID")
        
        # Optional fields
        validated['party'] = InputValidator.sanitize_string(data.get('party', ''), 100)
        validated['position'] = InputValidator.sanitize_string(data.get('position', 'Candidate'), 100)
        validated['candidate_image'] = data.get('candidate_image', '')
        validated['party_logo'] = data.get('party_logo', '')
        
        return validated
    
    @staticmethod
    def validate_vote_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate voting data"""
        validated = {}
        
        # Voter ID validation
        if not data.get('voter_id'):
            raise ValidationError("Voter ID is required")
        validated['voter_id'] = InputValidator.validate_voter_id(data['voter_id'])
        
        # Election ID validation
        if not data.get('election_id'):
            raise ValidationError("Election ID is required")
        try:
            validated['election_id'] = int(data['election_id'])
        except (ValueError, TypeError):
            raise ValidationError("Invalid election ID")
        
        # Candidate ID validation
        if not data.get('candidate_id'):
            raise ValidationError("Candidate ID is required")
        try:
            validated['candidate_id'] = int(data['candidate_id'])
        except (ValueError, TypeError):
            raise ValidationError("Invalid candidate ID")
        
        return validated
    
    @staticmethod
    def validate_registration_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete registration data"""
        validated = {}
        
        # Personal information
        validated['name'] = InputValidator.validate_name(data.get('name', ''))
        validated['address'] = InputValidator.validate_address(data.get('address', ''))
        validated['gender'] = InputValidator.validate_gender(data.get('gender', ''))
        validated['aadhaar'] = InputValidator.validate_aadhaar(data.get('aadhaar', ''))
        validated['phone'] = InputValidator.validate_phone(data.get('phone', ''), required=False)
        validated['email'] = InputValidator.validate_email(data.get('email', ''))
        validated['voter_id'] = InputValidator.validate_voter_id(data.get('voter_id', ''))
        
        # Date of birth and age validation
        dob, age = InputValidator.validate_date_of_birth(data.get('dob', ''))
        validated['dob'] = dob
        validated['age'] = age
        
        # Image validation
        validated['image_data_url'] = InputValidator.validate_image_data_url(data.get('image_data_url', ''))
        
        # Optional verification data
        validated['liveness_verified'] = bool(data.get('liveness_verified', False))
        validated['verification_type'] = InputValidator.sanitize_string(data.get('verification_type', 'basic_verification'), 50)
        validated['face_embedding_from_live'] = data.get('face_embedding_from_live')
        
        return validated
    
    @staticmethod
    def validate_login_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate login data"""
        validated = {}
        
        validated['voter_id'] = InputValidator.validate_voter_id(data.get('voter_id', ''))
        validated['email'] = InputValidator.validate_email(data.get('email', ''))
        validated['image_data_url'] = InputValidator.validate_image_data_url(data.get('image_data_url', ''))
        
        return validated

# Convenience functions for common validations
def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Check if all required fields are present"""
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")

def sanitize_dict(data: Dict[str, Any], max_length: int = 1000) -> Dict[str, Any]:
    """Sanitize all string values in a dictionary"""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = InputValidator.sanitize_string(value, max_length)
        else:
            sanitized[key] = value
    return sanitized