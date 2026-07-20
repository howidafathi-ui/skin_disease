# -*- coding: utf-8 -*-
from functools import wraps
from flask import abort
from flask_login import current_user


def role_required(role):
    """ديكوريتور للتحقق من أن المستخدم الحالي يملك الصلاحية المطلوبة (doctor / patient)."""
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                abort(403)
            return view_func(*args, **kwargs)
        return wrapped
    return decorator
