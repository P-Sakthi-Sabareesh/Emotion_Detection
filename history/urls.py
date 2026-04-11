from django.urls import path

from history import views

urlpatterns = [
    path("api/v1/records/<int:record_id>/", views.delete_record, name="api-delete-record"),
    path("api/v1/records/sha256/<str:sha256>/", views.delete_by_sha256, name="api-delete-sha256"),
    path("api/v1/records/retention/", views.retention_policy, name="api-retention"),
]
