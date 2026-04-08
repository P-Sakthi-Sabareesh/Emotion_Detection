from django.shortcuts import render


def home(request):
    return render(request, "core/home.html")


def prediction_hub(request):
    return render(request, "core/prediction_hub.html")


def about(request):
    return render(request, "core/about.html")


def contact(request):
    return render(request, "core/contact.html")

# Create your views here.
