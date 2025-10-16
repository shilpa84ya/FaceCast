"""
Security Middleware and Utilities for FaceCast
Provides rate limiting, CSRF protection, and security headers
"""

import time
import hashlib
import secrets
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, session, g
from typing import Dict, List, Optional, Callable, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger('facecast.security')

class RateLimiter:
    """Rate limiting implementation using sliding window"""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.blocked_ips = {}
    
    def is_allowed(self, identifier: str, max_requests: int = 100, window_minutes: int = 15) -> bool:
        """Check if request is allowed based on rate limits"""
        now = time.time()
        window_start = now - (window_minutes * 60)
        
        # Clean old requests
        while self.requests[identifier] and self.requests[identifier][0] < window_start:
            self.requests[identifier].popleft()
        
        # Check if IP is temporarily blocked
        if identifier in self.blocked_ips:
            if now < self.blocked_ips[identifier]:
                return False
            else:
                del self.blocked_ips[identifier]
        
        # Check rate limit
        if len(self.requests[identifier]) >= max_requests:
            # Block IP for 1 hour if rate limit exceeded
            self.blocked_ips[identifier] = now + 3600
            security_logger.warning(f"Rate limit exceeded for {identifier}. Blocked for 1 hour.")
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str, max_requests: int = 100, window_minutes: int = 15) -> int:
        """Get remaining requests for identifier"""
        now = time.time()
        window_start = now - (window_minutes * 60)
        
        # Clean old requests
        while self.requests[identifier] and self.requests[identifier][0] < window_start:
            self.requests[identifier].popleft()
        
        return max(0, max_requests - len(self.requests[identifier]))

class CSRFProtection:
    """CSRF token generation and validation"""
    
    @staticmethod
    def generate_token() -> str:
        """Generate a new CSRF token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_token(token: str) -> bool:
        """Validate CSRF token"""
        session_token = session.get('csrf_token')
        if not session_token or not token:
            return False
        return secrets.compare_digest(session_token, token)
    
    @staticmethod
    def set_token():
        """Set CSRF token in session"""
        if 'csrf_token' not in session:
            session['csrf_token'] = CSRFProtection.generate_token()

class SecurityHeaders:
    """Security headers management"""
    
    @staticmethod
    def add_security_headers(response):
        """Add security headers to response"""
        response.headers['X-Content-Type-Options'] = 'nosniff'\n        response.headers['X-Frame-Options'] = 'DENY'\n        response.headers['X-XSS-Protection'] = '1; mode=block'\n        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'\n        response.headers['Content-Security-Policy'] = (\n            \"default-src 'self'; \"\n            \"script-src 'self' 'unsafe-inline' 'unsafe-eval' https://fonts.googleapis.com; \"\n            \"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; \"\n            \"font-src 'self' https://fonts.gstatic.com; \"\n            \"img-src 'self' data: blob:; \"\n            \"media-src 'self' blob:; \"\n            \"connect-src 'self';\"\n        )\n        return response

class SessionSecurity:
    """Session security utilities"""
    
    @staticmethod\n    def is_session_valid() -> bool:\n        \"\"\"Check if current session is valid\"\"\"\n        if 'session_id' not in session:\n            return False\n        \n        # Check session timeout (2 hours)\n        if 'last_activity' in session:\n            last_activity = datetime.fromisoformat(session['last_activity'])\n            if datetime.now() - last_activity > timedelta(hours=2):\n                return False\n        \n        return True\n    \n    @staticmethod\n    def refresh_session():\n        \"\"\"Refresh session activity timestamp\"\"\"\n        session['last_activity'] = datetime.now().isoformat()\n        if 'session_id' not in session:\n            session['session_id'] = secrets.token_urlsafe(32)\n    \n    @staticmethod\n    def invalidate_session():\n        \"\"\"Invalidate current session\"\"\"\n        session.clear()\n\nclass AuditLogger:\n    \"\"\"Security audit logging\"\"\"\n    \n    @staticmethod\n    def log_security_event(event_type: str, details: Dict[str, Any], severity: str = 'INFO'):\n        \"\"\"Log security events\"\"\"\n        log_entry = {\n            'timestamp': datetime.now().isoformat(),\n            'event_type': event_type,\n            'ip_address': request.remote_addr if request else 'unknown',\n            'user_agent': request.headers.get('User-Agent', 'unknown') if request else 'unknown',\n            'details': details,\n            'severity': severity\n        }\n        \n        if severity == 'ERROR':\n            security_logger.error(f\"Security Event: {log_entry}\")\n        elif severity == 'WARNING':\n            security_logger.warning(f\"Security Event: {log_entry}\")\n        else:\n            security_logger.info(f\"Security Event: {log_entry}\")\n    \n    @staticmethod\n    def log_login_attempt(voter_id: str, success: bool, reason: str = ''):\n        \"\"\"Log login attempts\"\"\"\n        AuditLogger.log_security_event(\n            'LOGIN_ATTEMPT',\n            {\n                'voter_id': voter_id,\n                'success': success,\n                'reason': reason\n            },\n            'INFO' if success else 'WARNING'\n        )\n    \n    @staticmethod\n    def log_registration_attempt(voter_id: str, success: bool, reason: str = ''):\n        \"\"\"Log registration attempts\"\"\"\n        AuditLogger.log_security_event(\n            'REGISTRATION_ATTEMPT',\n            {\n                'voter_id': voter_id,\n                'success': success,\n                'reason': reason\n            },\n            'INFO' if success else 'WARNING'\n        )\n    \n    @staticmethod\n    def log_vote_cast(voter_id: str, election_id: int, candidate_id: int, success: bool):\n        \"\"\"Log vote casting\"\"\"\n        AuditLogger.log_security_event(\n            'VOTE_CAST',\n            {\n                'voter_id': voter_id,\n                'election_id': election_id,\n                'candidate_id': candidate_id,\n                'success': success\n            },\n            'INFO' if success else 'ERROR'\n        )\n    \n    @staticmethod\n    def log_admin_action(admin_id: str, action: str, details: Dict[str, Any]):\n        \"\"\"Log admin actions\"\"\"\n        AuditLogger.log_security_event(\n            'ADMIN_ACTION',\n            {\n                'admin_id': admin_id,\n                'action': action,\n                **details\n            },\n            'INFO'\n        )\n\n# Global instances\nrate_limiter = RateLimiter()\n\n# Decorators for security\ndef require_rate_limit(max_requests: int = 100, window_minutes: int = 15):\n    \"\"\"Decorator to enforce rate limiting\"\"\"\n    def decorator(f: Callable) -> Callable:\n        @wraps(f)\n        def decorated_function(*args, **kwargs):\n            identifier = request.remote_addr\n            \n            if not rate_limiter.is_allowed(identifier, max_requests, window_minutes):\n                AuditLogger.log_security_event(\n                    'RATE_LIMIT_EXCEEDED',\n                    {'ip': identifier, 'endpoint': request.endpoint},\n                    'WARNING'\n                )\n                return jsonify({\n                    'error': 'Rate limit exceeded. Please try again later.',\n                    'retry_after': 3600\n                }), 429\n            \n            return f(*args, **kwargs)\n        return decorated_function\n    return decorator\n\ndef require_csrf_token(f: Callable) -> Callable:\n    \"\"\"Decorator to require CSRF token for state-changing operations\"\"\"\n    @wraps(f)\n    def decorated_function(*args, **kwargs):\n        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:\n            token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')\n            if not CSRFProtection.validate_token(token):\n                AuditLogger.log_security_event(\n                    'CSRF_TOKEN_INVALID',\n                    {'endpoint': request.endpoint},\n                    'WARNING'\n                )\n                return jsonify({'error': 'Invalid CSRF token'}), 403\n        \n        return f(*args, **kwargs)\n    return decorated_function\n\ndef require_valid_session(f: Callable) -> Callable:\n    \"\"\"Decorator to require valid session\"\"\"\n    @wraps(f)\n    def decorated_function(*args, **kwargs):\n        if not SessionSecurity.is_session_valid():\n            return jsonify({'error': 'Invalid or expired session'}), 401\n        \n        SessionSecurity.refresh_session()\n        return f(*args, **kwargs)\n    return decorated_function\n\ndef log_endpoint_access(f: Callable) -> Callable:\n    \"\"\"Decorator to log endpoint access\"\"\"\n    @wraps(f)\n    def decorated_function(*args, **kwargs):\n        start_time = time.time()\n        \n        try:\n            result = f(*args, **kwargs)\n            \n            # Log successful access\n            AuditLogger.log_security_event(\n                'ENDPOINT_ACCESS',\n                {\n                    'endpoint': request.endpoint,\n                    'method': request.method,\n                    'duration_ms': round((time.time() - start_time) * 1000, 2),\n                    'status': 'success'\n                },\n                'INFO'\n            )\n            \n            return result\n            \n        except Exception as e:\n            # Log failed access\n            AuditLogger.log_security_event(\n                'ENDPOINT_ACCESS',\n                {\n                    'endpoint': request.endpoint,\n                    'method': request.method,\n                    'duration_ms': round((time.time() - start_time) * 1000, 2),\n                    'status': 'error',\n                    'error': str(e)\n                },\n                'ERROR'\n            )\n            raise\n    \n    return decorated_function\n\ndef sanitize_request_data(f: Callable) -> Callable:\n    \"\"\"Decorator to sanitize request data\"\"\"\n    @wraps(f)\n    def decorated_function(*args, **kwargs):\n        # Store original request data\n        g.original_json = request.get_json() if request.is_json else None\n        \n        return f(*args, **kwargs)\n    return decorated_function\n\n# Utility functions\ndef get_client_ip() -> str:\n    \"\"\"Get client IP address, considering proxies\"\"\"\n    if request.headers.get('X-Forwarded-For'):\n        return request.headers.get('X-Forwarded-For').split(',')[0].strip()\n    elif request.headers.get('X-Real-IP'):\n        return request.headers.get('X-Real-IP')\n    else:\n        return request.remote_addr\n\ndef hash_sensitive_data(data: str) -> str:\n    \"\"\"Hash sensitive data for logging\"\"\"\n    return hashlib.sha256(data.encode()).hexdigest()[:16]\n\ndef is_suspicious_request() -> bool:\n    \"\"\"Check if request appears suspicious\"\"\"\n    # Check for common attack patterns\n    suspicious_patterns = [\n        'script', 'javascript:', 'vbscript:', 'onload', 'onerror',\n        'eval(', 'alert(', 'document.cookie', 'window.location',\n        'union select', 'drop table', 'insert into', 'delete from',\n        '../', '..\\\\', '/etc/passwd', '/proc/version'\n    ]\n    \n    # Check URL and query parameters\n    full_url = request.url.lower()\n    for pattern in suspicious_patterns:\n        if pattern in full_url:\n            return True\n    \n    # Check request body if JSON\n    if request.is_json:\n        try:\n            json_str = str(request.get_json()).lower()\n            for pattern in suspicious_patterns:\n                if pattern in json_str:\n                    return True\n        except:\n            pass\n    \n    return False\n\ndef block_suspicious_requests(f: Callable) -> Callable:\n    \"\"\"Decorator to block suspicious requests\"\"\"\n    @wraps(f)\n    def decorated_function(*args, **kwargs):\n        if is_suspicious_request():\n            AuditLogger.log_security_event(\n                'SUSPICIOUS_REQUEST_BLOCKED',\n                {\n                    'url': request.url,\n                    'method': request.method,\n                    'user_agent': request.headers.get('User-Agent', 'unknown')\n                },\n                'WARNING'\n            )\n            return jsonify({'error': 'Request blocked for security reasons'}), 403\n        \n        return f(*args, **kwargs)\n    return decorated_function