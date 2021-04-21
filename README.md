# BirdWatcher
This is a project to build a bird sensing and identifying birdhouse.
The basic criteria that need to be satisfied are: 
<br><br>
<b>Software</b>
<ol>
  <li>Take video feed from camera, and sample images at either predetermined sample rate, or when a sensor is triggered;</li>
  <li>Run the image through pre-trained ConvNet model to determine bird species;</li>  
  <li>Push notification to something (undetermined, either app or just email/text) with picture of bird and identification;</li>
  <li>Analyze bird sounds to determine bird species (stretch);</li>
  <li>Database to store birds spotted per day (species, count, accuracy, etc.);</li>
</ol>
<b>Hardware</b>
<ol>
  <li>Camera that feeds to Rasberry Pi or similar;</li>
  <li>Rasberry Pi or similar with TPU if necessary;</li>  
  <li>Waterproof enclosure;</li>
  <li>Sensor to determine whether bird is feeding;</li>
  <li>Pre-built spillproof birdhouse;</li>
  <li>Power supply (potentially solar, but it is Newfoundland...);</li>
</ol>
