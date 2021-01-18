<script>
  let data;
  let currentTime = 0;
  let paused = true;
</script>

<video src="labeled.mp4" type="video/mp4" controls bind:currentTime bind:paused>
  <track kind="captions" />
</video>
