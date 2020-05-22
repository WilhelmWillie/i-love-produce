from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from .predict_vegetable import predict

app = Flask(__name__)

def process_media(URL, resp):
  # Add a message
  predict_response = predict(URL)
  resp.message(predict_response)
  return str(resp)

@app.route("/process_text", methods=['GET', 'POST'])
def handle_mms():
  # Get the message the user sent our Twilio number
  body = request.values.get('Body', None)

  num_media = int(request.values.get('NumMedia', 0))

  """Respond to incoming calls with a simple text message."""
  # Start our TwiML response
  resp = MessagingResponse()

  if num_media == 1:
    # Process single image 
    media_files = [(request.values.get("MediaUrl{}".format(i), ''),
                    request.values.get("MediaContentType{}".format(i), ''))
                    for i in range(0, num_media)]
    
    # Get first URL
    URL_TO_PROCESS = media_files[0][0]

    return process_media(URL_TO_PROCESS, resp)
  else:
    resp.message("Please add ONE image of a vegetable to your text!")
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)