document.addEventListener('DOMContentLoaded', () => {
    const electionStatusEl = document.getElementById('electionStatus');
    const myVotesEl = document.getElementById('myVotes');
    const liveResultsEl = document.getElementById('liveResults');
    const notificationsEl = document.getElementById('notifications');
    const userNameEl = document.getElementById('userName');
    const userImageEl = document.getElementById('userImage');
    const userEmailEl = document.getElementById('userEmail');
  
    async function getDashboardData() {
      try {
        // You'll need to handle user authentication here.
        // For now, we'll assume a dummy user.
        const response = await fetch('/admin/dashboard'); 
        const data = await response.json();
  
        if (response.ok) {
          electionStatusEl.textContent = data.electionStatus;
          myVotesEl.textContent = data.myVotes;
          liveResultsEl.textContent = data.liveResults;
          notificationsEl.textContent = data.notifications;
        } else {
          throw new Error(data.message || 'Failed to fetch dashboard data.');
        }
      } catch (error) {
        console.error('Error:', error);
        // Display a user-friendly error message on the cards
        electionStatusEl.textContent = 'Error loading data.';
        myVotesEl.textContent = 'Error loading data.';
        liveResultsEl.textContent = 'Error loading data.';
        notificationsEl.textContent = 'Error loading data.';
      }
    }
  
    // Fetch the data when the page loads
    getDashboardData();
  });