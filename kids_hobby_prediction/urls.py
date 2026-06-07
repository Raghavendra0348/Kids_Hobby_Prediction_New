"""
kids_hobby_prediction URL Configuration — Desktop-aware version

In desktop mode (served by Django):
  /          → serves React index.html  (SPA handles all frontend routes)
  /api/      → all REST endpoints
  /admin/    → Django admin
  /static/   → JS/CSS assets from the React build

In dev mode (Vite running separately):
  Django only answers /api/ and /admin/ — Vite handles everything else.
"""

from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.views.generic import TemplateView
from django.views.static import serve as static_serve
from django.template.response import TemplateResponse
from django.http import JsonResponse

def splash_view(request):
    """Splash page with no-cache headers so WebKit2 always loads fresh HTML."""
    response = TemplateResponse(request, 'splash.html')
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response

def health_check(request):
    """Health check endpoint — returns 200 OK so cron pingers get a green status."""
    return JsonResponse({'status': 'ok'})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('prediction.api_urls')),
    path('api/health/', health_check, name='health-check'),
    # Splash intro page — no-cache so changes show immediately
    path('splash/', splash_view, name='splash'),
]

# ── Desktop mode: serve the React build from Django ──────────────────────
# When DESKTOP_MODE=true (set by launcher.py), Django also serves:
#   /static/ → the React build's JS/CSS/assets
#   /*       → index.html so React Router handles the URL
import os
if os.environ.get('DESKTOP_MODE') == 'true':
    from django.conf.urls.static import static as _static_urls

    # Serve static assets from the React dist folder
    REACT_STATIC = getattr(settings, 'REACT_BUILD_STATIC', None)
    REACT_BUILD_DIR = getattr(settings, 'REACT_BUILD_DIR', None)

    if REACT_STATIC:
        urlpatterns += [
            re_path(
                r'^assets/(?P<path>.*)$',
                static_serve,
                {'document_root': REACT_STATIC},
            ),
        ]

    # Serve root-level public files from the React dist folder
    # e.g. /login_illustration.png, /favicon.svg, /icons.svg
    if REACT_BUILD_DIR and REACT_BUILD_DIR.exists():
        urlpatterns += [
            re_path(
                r'^(?P<path>.*\.(png|svg|ico|jpg|jpeg|gif|webp|json|txt))$',
                static_serve,
                {'document_root': str(REACT_BUILD_DIR)},
            ),
        ]

    # Explicit routes for React SPA pages (avoids PyWebView catch-all flicker)
    spa_view = TemplateView.as_view(template_name='index.html')
    urlpatterns += [
        path('login',    spa_view, name='react-login'),
        path('register', spa_view, name='react-register'),
        path('predict',  spa_view, name='react-predict'),
        path('profile',  spa_view, name='react-profile'),
        path('history',  spa_view, name='react-history'),
        path('about',    spa_view, name='react-about'),
        path('contact',  spa_view, name='react-contact'),
        path('dashboard',spa_view, name='react-dashboard'),
    ]

    # Catch-all: any remaining URL returns index.html
    urlpatterns += [
        re_path(r'^(?!api/|admin/|static/|assets/|splash/).*$',
                spa_view,
                name='react-spa'),
    ]
