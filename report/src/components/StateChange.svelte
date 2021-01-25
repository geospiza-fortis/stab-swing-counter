<script>
  import { zip, chunk } from "lodash";
  import { currentTime, paused } from "../store.js";

  export let data;
  const keys = [
    "label",
    "start",
    "end",
    "diff",
    "stab",
    "swing",
    "total",
    "stab_pct",
    "swing_pct"
  ];

  export let rowId = 0;
  export let paginationSize = 6;
  $: labelData = data ? transform(data) : [];
  $: foundIndex = labelData.findIndex(
    row => row.start >= Math.floor($currentTime * 60)
  );
  $: rowId = foundIndex >= 0 ? foundIndex : labelData.length - 1;
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
      if (e == value) {
        continue;
      }
      if (value != "other") {
        res.push({
          pos: res.length,
          label: value,
          start: start,
          end: i,
          // time since the last attack
          diff: res.length > 0 ? start - res[res.length - 1].start : 0,
          stab: res.filter(x => x.label == "stab").length + (value == "stab"),
          swing:
            res.filter(x => x.label == "swing").length + (value == "swing"),
          total: res.length + 1
        });
      }
      start = i;
      value = e;
    }
    res = res.map(row => ({
      ...row,
      stab_pct: (row.stab / row.total).toFixed(2),
      swing_pct: (row.swing / row.total).toFixed(2)
    }));
    return res;
  }
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
            $paused = true;
            $currentTime = row.start / 60;
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
