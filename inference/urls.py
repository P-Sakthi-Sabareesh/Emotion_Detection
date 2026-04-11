from django.urls import path

from inference import views

urlpatterns = [
    path("prediction/upload/", views.upload_prediction_page, name="prediction-upload"),
    path("prediction/live/", views.live_prediction_page, name="prediction-live"),
    # Health
    path("healthz", views.healthz, name="api-healthz"),
    path("readyz", views.readyz, name="api-readyz"),
    # Legacy alias kept for existing clients.
    path("api/health/", views.healthz, name="api-health"),
    # Prediction APIs
    path("api/v1/predict/image/", views.predict_image_api, name="api-predict-image"),
    path("api/v1/predict/live-frame/", views.predict_live_frame_api, name="api-predict-live"),
    path(
        "api/v1/predict/sample/<str:sample_name>/",
        views.predict_sample_api,
        name="api-predict-sample",
    ),
    # Unversioned paths kept to avoid breaking existing clients on upgrade.
    path("api/predict/image/", views.predict_image_api),
    path("api/predict/live-frame/", views.predict_live_frame_api),
    path("api/predict/sample/<str:sample_name>/", views.predict_sample_api),
]
