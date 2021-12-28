from django import forms
from cnn.models import ImageTest

class ImageTestForm(forms.ModelForm):
    class Meta:
        model = ImageTest
        fields = '__all__'