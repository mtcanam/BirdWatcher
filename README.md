# BirdWatcher
This is a project to build a bird sensing and identifying birdhouse.
The basic criteria that need to be satisfied are: 
<br><br>
<b>Software</b>
<ul>
  <li>Take video feed from camera, and sample images at either predetermined sample rate, or when a sensor is triggered;</li>
  <li>Run the image through pre-trained ConvNet model to determine bird species;</li>  
  <li>Push notification to something (undetermined, either app or just email/text) with picture of bird and identification;</li>
  <li>Analyze bird sounds to determine bird species (stretch);</li>
  <li>Database to store birds spotted per day (species, count, accuracy, etc.);</li>
</ul>
<b>Hardware</b>
<ol>
  <li>Camera that feeds to Rasberry Pi or similar;</li>
  <li>Rasberry Pi or similar with TPU if necessary;</li>  
  <li>Waterproof enclosure;</li>
  <li>Sensor to determine whether bird is feeding;</li>
  <li>Pre-built spillproof birdhouse;</li>
  <li>Power supply (potentially solar, but it is Newfoundland...);</li>
</ol>

<b>Workflows</b>

Video Workflow
<ol>
  <li>Bird is sensed</li>
  <li>Still is taken from camera and fed to bird-video-identifier</li>
  <li>Return is species, probability</li>
  <li>Data is fed to graphQL database and to notification service</li>
</ol>

Audio Workflow
<ol>
  <li>Microphone detects birdsong</li>
  <li>Clip is taken from camera and fed to bird-audio-identifier</li>
  <li>Return is species, probability</li>
  <li>Data is fed to graphQL database and to notification service</li>
</ol>
