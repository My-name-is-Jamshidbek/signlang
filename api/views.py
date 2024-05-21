from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import numpy as np
from PIL import Image
from core.settings import runnsoz, runnharf

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request):
    # Check if the image is provided in the request
    if 'image' not in request.FILES:
        return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Read the image file from the request
        image_file = request.FILES['image']
        image = Image.open(image_file)
        image = np.array(image)

        # Process the image to text using runnharf
        txt = runnharf.frame_to_txt(image)
        if txt:
            txt = txt[0]
        else:
            txt = "No text"
        print(txt)
        # Return the processed text in the response
        return Response(txt, status=status.HTTP_200_OK)

    except Exception as e:
        # Handle exceptions and return an error response
        print(e)
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
