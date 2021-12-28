from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponseRedirect
from cnn.models import ImageTest
from cnn.forms import ImageTestForm
from cnn.scripts.CNN import model
import os
import cv2
import numpy as np

from ImageRetrieval.settings import BASE_DIR

from django.views.generic.edit import CreateView

def create_image(request):
    
    if request.method == 'POST':
        form = ImageTestForm(request.POST, request.FILES)
        if form.is_valid() and form.cleaned_data['image']:
            image = ImageTest(image=form.cleaned_data['image'])
            image.save()

            img_path = r'uploads/' + image.image.name
            img_data = []

            img_path = os.path.join(BASE_DIR, img_path)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (150,150))
            img_data.append(img)

            img_data = np.array(img_data)

            test_val = img_data.astype('float32') / 255.0

            pred = model.predict(test_val)
            labels = (pred > 0.5).astype(np.int)

            image.delete()
            os.remove(img_path)

            if(labels[0][0] == 0):
                picture_type = "Gambar adalah daun Jambu biji"
            else:
                picture_type = "Gambar adalah daun Pandan"

            context = {
            'form': form,
            'picture_type': picture_type
            }
            
            return render(request, 'index.html', context)
    else:
        form = ImageTestForm()

    context = {
        'form': form,
        'picture_type': "none"
    }

    return render(request, 'index.html', context)

def create_image_success(request):
    context = {
        'success': 'nothing',
    }

    return render(request, 'success.html', context)

class ImageCreate(CreateView):
    model = ImageTest
    form_class = ImageTestForm