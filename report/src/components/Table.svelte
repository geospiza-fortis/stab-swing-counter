<script>
  export let data;
</script>

<table class="table table-sm table-bordered">
  <thead>
    {#each Object.keys(data[0]) as key}
      <th>{key}</th>
    {/each}
  </thead>
  <tbody>
    {#each data as row, i}
      <tr class={data.length == i + 1 ? 'table-primary' : ''}>
        {#each Object.keys(data[0]) as key}
          <td>{row[key]}</td>
        {/each}
      </tr>
    {/each}
  </tbody>
</table>
