from django.urls import path

from core import views

urlpatterns = [
    path("", views.home, name="home"),
    path("prediction/", views.prediction_hub, name="prediction-hub"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
]
