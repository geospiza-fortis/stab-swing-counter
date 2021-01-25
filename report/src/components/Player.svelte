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

<h2>Analysis for trial data</h2>

<p>
  Click on the video to begin. Clicking on the plot or table will bring the
  video to that frame. Classification confidence is calculated by a
  <a
    href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV">
    Ridge classifier
  </a>
  trained on
  <a
    href="https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html">
    template matching intensities
  </a>
  over a window of 4 frames.
</p>

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

  <h3 style="text-align:center;">Trial {trialId}</h3>

  {#await fetchText(`trial/${trialId}/pred.csv`) then data}
    <Plot {data} />
  {/await}

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
        {#await fetchJson(`trial/${trialId}/pred.json`) then data}
          <StateChange {data} />
        {/await}
      </div>
    </div>
  </div>

{/await}
