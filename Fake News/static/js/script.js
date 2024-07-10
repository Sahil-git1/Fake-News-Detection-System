document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    // Theme toggle
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('light-theme');
    });

    // Add stars
    for (let i = 0; i < 100; i++) {
        const star = document.createElement('div');
        star.classList.add('star');
        star.style.left = `${Math.random() * 100}%`;
        star.style.top = `${Math.random() * 100}%`;
        star.style.animationDelay = `${Math.random() * 4}s`;
        body.appendChild(star);
    }

    // Add shooting stars
    setInterval(() => {
        const shootingStar = document.createElement('div');
        shootingStar.classList.add('shooting-star');
        shootingStar.style.left = `${Math.random() * 100}%`;
        shootingStar.style.top = `${Math.random() * 100}%`;
        body.appendChild(shootingStar);

        setTimeout(() => {
            shootingStar.remove();
        }, 5000);
    }, 10000);

    // Fake news check
    const checkBtn = document.getElementById('check-btn');
    const newsInput = document.getElementById('news-input');
    const result = document.getElementById('result');

    checkBtn.addEventListener('click', async () => {
        const newsText = newsInput.value;
        if (newsText.trim() === '') {
            result.textContent = 'Please enter some text to check.';
            return;
        }

        try {
            const response = await fetch('/check_news', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: newsText }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            result.textContent = data.result === 'fake' ? 'This news is likely fake.' : 'This news appears to be authentic.';
        } catch (error) {
            console.error('Error:', error);
            result.textContent = 'An error occurred while checking the news.';
        }
    });

    // Add space-themed background
    function addStars() {
        for (let i = 0; i < 100; i++) {
            const star = document.createElement('div');
            star.classList.add('star');
            star.style.left = `${Math.random() * 100}%`;
            star.style.top = `${Math.random() * 100}%`;
            star.style.animationDelay = `${Math.random() * 2}s`;
            body.appendChild(star);
        }
    }

    addStars();
});