from django.urls import path

from inference import views

urlpatterns = [
    path("prediction/upload/", views.upload_prediction_page, name="prediction-upload"),
    path("prediction/live/", views.live_prediction_page, name="prediction-live"),
    path("api/health/", views.health_api, name="api-health"),
    path("api/predict/image/", views.predict_image_api, name="api-predict-image"),
    path("api/predict/live-frame/", views.predict_live_frame_api, name="api-predict-live"),
    path("api/predict/sample/<str:sample_name>/", views.predict_sample_api, name="api-predict-sample"),
]
