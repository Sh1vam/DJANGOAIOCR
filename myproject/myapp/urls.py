from django.urls import path
from .views import UploadExtractImageView, GetExtractedDataView

urlpatterns = [
    path('upload-image/', UploadExtractImageView.as_view(), name='upload-image'),
    path('get-extracted-data/', GetExtractedDataView.as_view(), name='get-extracted-data'),
]