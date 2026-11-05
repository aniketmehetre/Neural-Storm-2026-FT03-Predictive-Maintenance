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

    // Split text cleanly by words, then characters to prevent unnatural word breaking
    const text = chapter.headline || '';
    const words = text.split(' ');
    this.titleEl.innerHTML = '';
    const letters = [];

    words.forEach((word, wIdx) => {
      const wordSpan = document.createElement('span');
      wordSpan.className = 'title-word';
      wordSpan.style.display = 'inline-block';
      wordSpan.style.whiteSpace = 'nowrap';

      word.split('').forEach((char) => {
        const charSpan = document.createElement('span');
        charSpan.className = 'title-char';
        charSpan.textContent = char;
        charSpan.style.display = 'inline-block';
        charSpan.style.opacity = '0';
        charSpan.style.transform = 'translateY(20px)';
        wordSpan.appendChild(charSpan);
        letters.push(charSpan);
      });

      this.titleEl.appendChild(wordSpan);

      // Normal whitespace between words so browser wraps cleanly between words
      if (wIdx < words.length - 1) {
        const space = document.createTextNode(' ');
        this.titleEl.appendChild(space);
      }
    });

    this.descEl.textContent = chapter.desc;
    this.descEl.style.opacity = '0';
    this.descEl.style.transform = 'translateY(10px)';

    // Anime.js v4 character reveal animation with refined cubic easing
    animate(letters, {
      opacity: [0, 1],
      translateY: [20, 0],
      delay: stagger(15),
      duration: 400,
      ease: 'outExpo',
    });

    animate(this.descEl, {
      opacity: [0, 1],
      translateY: [10, 0],
      duration: 450,
      delay: 120,
      ease: 'outQuad',
    });
  }
}

