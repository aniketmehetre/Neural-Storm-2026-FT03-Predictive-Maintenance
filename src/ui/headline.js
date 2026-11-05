/**
 * Headline: Kinetic chapter title swapped via anime.js text-split animations.
 */
import { animate, stagger } from 'animejs';

export class ChapterHeadline {
  constructor() {
    this.badgeEl = document.getElementById('chapter-badge');
    this.titleEl = document.getElementById('chapter-headline');
    this.descEl = document.getElementById('chapter-desc');
    this.currentChapterId = null;
  }

  update(chapter) {
    if (this.currentChapterId === chapter.id) return;
    this.currentChapterId = chapter.id;

    // Badge update
    this.badgeEl.textContent = chapter.badge;
    if (chapter.health > 0.6) {
      this.badgeEl.style.color = '#3de0ff';
    } else if (chapter.health > 0.3) {
      this.badgeEl.style.color = '#ffb23d';
    } else {
      this.badgeEl.style.color = '#ff4d4d';
    }

    // Split text into spans for staggered character reveal
    const text = chapter.headline;
    this.titleEl.innerHTML = '';
    const letters = [];

    text.split('').forEach((char) => {
      const span = document.createElement('span');
      span.textContent = char === ' ' ? '\u00A0' : char;
      span.style.display = 'inline-block';
      span.style.opacity = '0';
      span.style.transform = 'translateY(24px)';
      this.titleEl.appendChild(span);
      letters.push(span);
    });

    this.descEl.textContent = chapter.desc;
    this.descEl.style.opacity = '0';
    this.descEl.style.transform = 'translateY(12px)';

    // Anime.js v4 character reveal animation
    animate(letters, {
      opacity: [0, 1],
      translateY: [24, 0],
      delay: stagger(18),
      duration: 450,
      ease: 'outExpo',
    });

    animate(this.descEl, {
      opacity: [0, 1],
      translateY: [12, 0],
      duration: 500,
      delay: 150,
      ease: 'outQuad',
    });
  }
}

