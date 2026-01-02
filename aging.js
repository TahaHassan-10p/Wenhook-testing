/**
 * Calculate the total number of working days (Mon-Fri) a requisition has been "open".
 * @param created_at Date the requisition was created
 * @param statusHistory Array of { status: string, changed_at: Date }
 * @param currentStatus Current status of the requisition
 * @returns number of working days requisition has been "open"
 */
function calculateRequisitionAging(created_at, statusHistory, currentStatus) {
  // Helper: count working days between two dates (inclusive of start, exclusive of end)
  function countWorkingDays(start, end) {
    let count = 0;
    let d = new Date(start);
    while (d < end) {
      const day = d.getDay();
      if (day !== 0 && day !== 6) count++; // 0 = Sunday, 6 = Saturday
      d.setDate(d.getDate() + 1);
    }
    return count;
  }

  // Sort statusHistory by changed_at ascending
  const history = [...statusHistory].sort(
    (a, b) => new Date(a.changed_at) - new Date(b.changed_at)
  );

  let total = 0;
  let openStart = created_at;
  let lastStatus = "open";

  for (const entry of history) {
    const status = entry.status.toLowerCase();
    const changedAt = new Date(entry.changed_at);

    if (
      lastStatus === "open" &&
      (status === "on-hold" || status === "closed")
    ) {
      // Count working days from openStart to changedAt
      if (openStart) {
        total += countWorkingDays(openStart, changedAt);
      }
      openStart = null;
    }
    if (
      (lastStatus === "on-hold" || lastStatus === "closed") &&
      status === "open"
    ) {
      openStart = changedAt;
    }
    lastStatus = status;
  }

  // If currently open, count up to today
  if (lastStatus === "open" && openStart) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    total += countWorkingDays(openStart, today);
  }

  return total;
}

module.exports = { calculateRequisitionAging };
