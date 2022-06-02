Due to certain reason (Lack of documentation, or rather, hard to read) from AI Platform
We decide to create a custom backend to serve the ML Model

Also, due to how keras load it's model on the fly, we can't really use Docker (Cloud RUN) 
and must have a filesystem, so, unfortunately Compute Engine it is