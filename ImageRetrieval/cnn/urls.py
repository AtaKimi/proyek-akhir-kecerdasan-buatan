from django.urls import path
from cnn.views import create_image, create_image_success, ImageCreate
urlpatterns = [
    path('', create_image, name='create-image'),
    path('success/', create_image_success, name='success')
]