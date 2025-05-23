"""
WSGI config for the uci_phonotactic_calculator web interface.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "uci_phonotactic_calculator.web.django.settings"
)

application = get_wsgi_application()
