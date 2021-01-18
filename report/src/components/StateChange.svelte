<script>
  import { zip, chunk } from "lodash";
  import { onMount } from "svelte";
  import { currentTime, paused } from "../store.js";

  let labelData = [];
  const keys = ["pos", "label", "start", "end", "duration", "stab_pct"];

  export let rowId = 0;
  export let paginationSize = 6;
  $: rowId = labelData.findIndex(row => row.start > $currentTime * 60);
  $: idx = rowId ? Math.floor(rowId / paginationSize) : 0;
  $: chunked = chunk(labelData, paginationSize);
  $: total = labelData.length;
  $: pages = chunked.length;

  function prev() {
    if (idx > 0) {
      idx--;
    }
  }
  function next() {
    if (idx < pages - 1) {
      idx++;
    }
  }

  function transform(data) {
    // What frame did it start, and what frame did it end.
    let res = [];
    let start = 0;
    let value = "other";
    for (let i = 0; i < data.length; i++) {
      let e = data[i];
      if (e != value) {
        if (value != "other") {
          res.push({
            pos: res.length,
            label: value,
            start: start,
            end: i,
            // time since the last attack
            duration: res.length > 0 ? start - res[res.length - 1].start : 0,
            stab_pct: (
              (res.filter(x => x.label == "stab").length + (value == "stab")) /
              (res.length + 1)
            ).toFixed(2)
          });
        }
        start = i;
        value = e;
      }
    }
    return res;
  }

  onMount(async () => {
    let resp = await fetch("pred.json");
    let data = await resp.json();
    labelData = transform(data);
  });
</script>

{#if pages}
  <table class="table table-sm table-bordered">
    <thead>
      {#each keys as key}
        <th>{key}</th>
      {/each}
    </thead>
    <tbody>
      {#each chunked[idx] as row}
        <tr
          on:click={() => {
            rowId = row.pos;
            $paused = true;
            $currentTime = (row.start - 1) / 60;
          }}
          class={rowId == row.pos ? 'table-active' : ''}>
          {#each keys as key}
            <td>{row[key]}</td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
  <div class="btn-group" role="group">
    <button
      type="button"
      class="btn btn-outline-primary"
      disabled={!(idx > 0)}
      on:click={prev}>
      Prev
    </button>
    <button
      type="button"
      class="btn btn-outline-primary"
      disabled={!(idx < pages - 1)}
      on:click={next}>
      Next
    </button>
  </div>
  <div>
    <p>{total} rows - page {idx + 1} of {pages}</p>
  </div>
{/if}
