<script>
  import Video from "./Video.svelte";
  import Plot from "./Plot.svelte";
  import StateChange from "./StateChange.svelte";

  async function fetchManifest() {
    let data = await fetch("trial/manifest.json");
    return await data.json();
  }
  let trialId = "00";
  let controls = true;

  async function fetchText(src) {
    let resp = await fetch(src);
    return await resp.text();
  }

  async function fetchJson(src) {
    let resp = await fetch(src);
    return await resp.json();
  }
</script>

{#await fetchManifest() then manifest}
  <nav>
    <ul class="nav">
      {#each manifest as trial}
        <li class="nav-item">
          <a
            class="nav-link active"
            href="javascript:void(0)"
            on:click={() => (trialId = trial)}>
            trial {trial}
          </a>
        </li>
      {/each}
    </ul>
  </nav>

  <h2>Data for trial {trialId}</h2>
  <Plot src={`trial/${trialId}/pred.csv`} />

  <div class="container">
    <div class="row">
      <div class="col" style="text-align: center">
        <Video
          {controls}
          src={`https://storage.googleapis.com/geospiza/stab-swing-counter/v1/output/${trialId}/output.mp4`} />
        <label>
          <input type="checkbox" bind:checked={controls} />
          Show media controls
        </label>
      </div>
      <div class="col">
        <StateChange src={`trial/${trialId}/pred.json`} />
      </div>
    </div>
  </div>

{/await}
