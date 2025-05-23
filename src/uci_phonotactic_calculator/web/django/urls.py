"""URL patterns for the uci_phonotactic_calculator web interface."""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("uci_phonotactic_calculator.web.django.webcalc.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
