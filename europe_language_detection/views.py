from django.shortcuts import render
from django.http import JsonResponse
import pickle
labels = {
'bg': "Bulgarian",
'cs': "Czech",
'da': "Danish",
'de': "German",
'el': "Greek, Modern",
'en': "English",
'es': "Spanish",
'et': "Estonian",
'fi': "Finnish",
'fr': "French",
'hu': "Hungarian",
'it': "Italian",
'lt': "Lithuanian",
'lv': "Latvian",
'nl': "Dutch",
'pl': "Polish",
'pt': "Portuguese",
'ro': "Romanian",
'sk': "Slovak",
'sl': "Slovenian",
'sv': "Swedish",
}
def load():
    with open(r'C:\Users\ajain\Desktop\europe_language_detection\europe_language_detection\model.pkl', 'rb') as file:
      vectorizer, clf = pickle.load(file)
    return vectorizer, clf

vectorizer, classifer = load()

def prediction(request):
    ans = ''
    if request.method == 'POST':
        if request.is_ajax():
            email = request.POST.get('email')
            email_input = [email]
            email_input_transformed = vectorizer.transform(email_input)
            prediction = classifer.predict(email_input_transformed)
            for i, j in labels.items():
                if i == prediction:
                    ans = j
            data = {"email": ans}
            return JsonResponse(data)
    return render(request,'prediction.html')