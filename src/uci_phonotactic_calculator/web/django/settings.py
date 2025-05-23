"""
Django settings for uci_phonotactic_calculator web interface.

Adapted from the original webcalc_project settings.
"""

import os
from pathlib import Path

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Quick-start development settings - unsuitable for production
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "lfy!v6$$p_6ea(942x1&rn5)=v9t%7x1#$umf-yi(h=otq*%y0"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["127.0.0.1", "localhost", "phonotactics.socsci.uci.edu"]

X_FRAME_OPTIONS = "SAMEORIGIN"

# Application definition
INSTALLED_APPS = [
    "uci_phonotactic_calculator.web.django.webcalc.apps.WebcalcConfig",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_crontab",
]

# Use package-level media directory instead of top-level
MEDIA_ROOT = os.path.join(Path(__file__).resolve().parent, "media")
MEDIA_URL = "/media/"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "uci_phonotactic_calculator.web.django.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(Path(__file__).resolve().parent, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "uci_phonotactic_calculator.web.django.wsgi.application"

# Database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(Path(__file__).resolve().parent, "db.sqlite3"),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
# Use package-level static directory instead of top-level
STATIC_ROOT = os.path.join(Path(__file__).resolve().parent, "static/")

# Cronjobs for cleaning media files
CRONJOBS = [
    (
        "*/10 * * * *",
        "uci_phonotactic_calculator.web.django.webcalc.cron.clean_media_folder",
        ">> " + os.path.join(Path(__file__).resolve().parent, "logs/clean_media_cron.log" + " 2>&1 "),
    )
]

SILENCED_SYSTEM_CHECKS = ["models.W042"]
