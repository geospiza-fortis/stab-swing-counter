<script>
  import { currentTime, paused } from "../store.js";
</script>

<video
  src="labeled.mp4"
  type="video/mp4"
  controls
  bind:currentTime={$currentTime}
  bind:paused={$paused}>
  <track kind="captions" />
</video>
