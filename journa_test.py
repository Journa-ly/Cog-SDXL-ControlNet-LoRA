import os
from dotenv import load_dotenv
import replicate

load_dotenv()


deployment = replicate.deployments.get("nick-harvey/journahelloworld")
prediction = deployment.predictions.create(
  input={"prompt": "An astronaut riding a rainbow unicorn"}
)
prediction.wait()
print(prediction.output)
