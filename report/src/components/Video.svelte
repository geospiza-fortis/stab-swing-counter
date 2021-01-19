<script>
  import { currentTime, paused } from "../store.js";

  export let src = "labeled.mp4";
</script>

<video
  {src}
  type="video/mp4"
  controls
  webkit-playsinline
  playsinline
  bind:currentTime={$currentTime}
  bind:paused={$paused}>
  <track kind="captions" />
</video>
