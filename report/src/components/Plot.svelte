<script>
  import { zip } from "lodash";
  import Papa from "papaparse";
  import { onMount } from "svelte";
  import { currentTime, paused } from "../store.js";

  export let src = "pred.csv";
  let predData;
  let plotElement;
  let plot;
  let range = [0, 1200];
  let frame = 0;

  // we have to make sure we're in a consistent state when updating the range,
  // otherwise fall prey to infinite loops.
  $: frame = Math.floor($currentTime * 60);
  $: !$paused && plot && updateRange(plotElement, range, frame);

  function updateRange(element, range, frame) {
    let size = range[1] - range[0];
    // keep the current frame in the middle of the window
    let x0 = frame - size / 2;
    let x1 = frame + size / 2;
    let newRange = [x0, x1];

    // must change by a certain percentage before moving the slider
    if (Math.abs(range[0] - x0) / size < 0.3) {
      return;
    }

    Plotly.relayout(element, "xaxis.range", newRange);
  }

  $: predData &&
    plotElement &&
    Plotly.Fx.hover(
      plotElement,
      [0, 1, 2].map(trace => ({
        curveNumber: trace,
        pointNumber: frame
      }))
    );

  function transform(data, args) {
    return zip(...data).map((row, i) => ({ y: row, ...args[i] }));
  }

  onMount(async () => {
    let resp = await fetch(src);
    let data = await resp.text();
    predData = Papa.parse(data).data;
    plot = new Plotly.newPlot(
      plotElement,
      transform(predData, [
        { name: "other" },
        { name: "stab" },
        { name: "swing" }
      ]),
      {
        title: "Probability of being in an animation state",
        xaxis: {
          range: range,
          rangeselector: { visible: true },
          rangeslider: {}
        },
        margin: {
          l: 50,
          r: 0,
          b: 50,
          t: 50
        }
      },
      { responsive: true }
    );
    plotElement.on("plotly_relayout", ev => {
      let newRange = ev["xaxis.range"];
      if (!newRange) {
        return;
      }
      range = newRange;
    });
    plotElement.on("plotly_click", ev => {
      // grab an arbitrary trace
      let trace = ev.points[0];
      $currentTime = (trace.pointNumber - 1) / 60;
      $paused = true;
    });
  });
</script>

<div bind:this={plotElement} />
