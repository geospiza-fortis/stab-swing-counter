<script>
  import { zip } from "lodash";
  import Papa from "papaparse";
  import { onMount } from "svelte";

  let predData;
  let plotElement;

  function transform(data, args) {
    return zip(...data).map((row, i) => ({ y: row, ...args[i] }));
  }

  onMount(async () => {
    let resp = await fetch("pred.csv");
    let data = await resp.text();
    predData = Papa.parse(data).data;
    let plot = new Plotly.newPlot(
      plotElement,
      transform(predData, [
        { name: "other" },
        { name: "stab" },
        { name: "swing" }
      ]),
      {
        title: "Probability of being in an animation state",
        xaxis: {
          range: [0, 1200],
          rangeselector: { visible: true },
          rangeslider: {}
        },
        margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
        }
      }
    );
  });
</script>

<div bind:this={plotElement} />
