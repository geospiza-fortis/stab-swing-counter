<script>
  import { zip } from "lodash";
  import Papa from "papaparse";
  import { onMount } from "svelte";

  let predData;
  let plotElement;

  function transform(data) {
    return zip(...data).map(row => ({ y: row }));
  }

  onMount(async () => {
    let resp = await fetch("pred.csv");
    let data = await resp.text();
    predData = Papa.parse(data).data;
    let plot = new Plotly.newPlot(plotElement, transform(predData), {
      xaxis: {
        rangeselector: { visible: true },
        rangeslider: {}
      },
      margin: {
        l: 50,
        r: 0,
        b: 50
      }
    });
  });
</script>

<style>
  main {
    max-width: 900px;
    margin: 0 auto;
  }
</style>

<main>
  <h1>Counting swings and stabs</h1>

  <div bind:this={plotElement} />

  {#if predData}{JSON.stringify(transform(predData))}{/if}
</main>
