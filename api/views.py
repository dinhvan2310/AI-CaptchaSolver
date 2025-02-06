from rest_framework.decorators import api_view
from rest_framework.response import Response
from utils.common import is_valid_url
from utils.captcha import resolver_captcha
from urllib.parse import unquote

@api_view(["POST"])
def resolverCaptchaView(request):
    url = request.data.get('url')
    if not url:
        return Response({'error': 'URL is required'}, status=400)
    if not is_valid_url(url):
        return Response({'error': 'Invalid URL'}, status=400)
    return Response(resolver_captcha(url))