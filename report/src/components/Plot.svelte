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
        range: [0, 1200],
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

<div bind:this={plotElement} />
