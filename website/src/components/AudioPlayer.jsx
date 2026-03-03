import { useState, useRef, useEffect } from 'react';

export default function AudioPlayer({ src, title = 'Audio Overview', demoId }) {
  const audioRef = useRef(null);
  const animFrameRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [activeSrc, setActiveSrc] = useState(src || null);

  const formatTime = (time) => {
    if (!time || isNaN(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handlePlayPause = () => {
    if (!isLoaded || hasError) return;
    if (isPlaying) {
      audioRef.current?.pause();
      setIsPlaying(false);
    } else {
      audioRef.current?.play();
      setIsPlaying(true);
    }
  };

  // Use requestAnimationFrame for smooth progress
  const updateProgress = useCallback(() => {
    if (audioRef.current && isPlaying) {
      setCurrentTime(audioRef.current.currentTime);
      animFrameRef.current = requestAnimationFrame(updateProgress);
    }
  }, [isPlaying]);

  useEffect(() => {
    if (isPlaying) {
      animFrameRef.current = requestAnimationFrame(updateProgress);
    }
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, updateProgress]);

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
      setIsLoaded(true);
    }
  };

  useEffect(() => {
    setActiveSrc(src || null);
    setHasError(false);
    setIsLoaded(false);
  }, [src]);

  const handleError = () => {
    setHasError(true);
    setIsLoaded(false);
  };

  const handleProgressClick = (e) => {
    if (!isLoaded || !audioRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audioRef.current.currentTime = percent * duration;
    setCurrentTime(percent * duration);
  };

  const handleEnded = () => setIsPlaying(false);

  useEffect(() => {
    return () => { audioRef.current?.pause(); };
  }, []);

  const showComingSoon = hasError || !src;
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Generate waveform bars (deterministic from demoId)
  const seed = demoId ? demoId.split('').reduce((a, c) => a + c.charCodeAt(0), 0) : 42;
  const bars = Array.from({ length: 48 }, (_, i) => {
    const h = Math.sin(i * 0.7 + seed * 0.1) * 0.3 + Math.sin(i * 1.3 + seed * 0.05) * 0.2 + 0.5;
    return Math.max(0.15, Math.min(1, h));
  });

  return (
    <div
      style={{
        position: 'relative',
        background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.8), rgba(22, 33, 62, 0.6))',
        border: '1px solid rgba(228, 187, 124, 0.12)',
        borderRadius: 8,
        padding: '28px 32px',
        transition: 'all 0.3s ease',
        overflow: 'hidden',
        ...(isHovered && !showComingSoon ? {
          borderColor: 'rgba(228, 187, 124, 0.25)',
          boxShadow: '0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(228,187,124,0.08)'
        } : {})
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Top accent line */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: 'linear-gradient(90deg, transparent 5%, var(--gold) 50%, transparent 95%)',
        opacity: showComingSoon ? 0.1 : (isHovered ? 0.5 : 0.25),
        transition: 'opacity 0.3s ease'
      }} />

      {showComingSoon ? (
        /* ---- Coming Soon State ---- */
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <div style={{
            width: 52, height: 52, borderRadius: '50%', flexShrink: 0,
            background: 'rgba(228, 187, 124, 0.06)',
            border: '1px solid rgba(228, 187, 124, 0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 22
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgba(228,187,124,0.35)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 18V5l12-2v13"/>
              <circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
            </svg>
          </div>
          <div style={{ flex: 1 }}>
            <h3 style={{
              fontFamily: 'var(--font-serif)', fontSize: 16, fontWeight: 400,
              color: 'rgba(228, 187, 124, 0.5)', margin: '0 0 4px 0', letterSpacing: '0.3px'
            }}>
              {title}
            </h3>
            <p style={{
              fontSize: 13, color: 'var(--text-muted)', margin: 0,
              opacity: 0.5, fontStyle: 'italic'
            }}>
              AI-generated audio discussion — coming soon
            </p>
          </div>
          {/* Static ghost waveform */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 2, opacity: 0.15, height: 32 }}>
            {bars.filter((_, i) => i % 2 === 0).map((h, i) => (
              <div key={i} style={{
                width: 2, height: `${h * 28}px`, borderRadius: 1,
                background: 'var(--gold)'
              }} />
            ))}
          </div>
        </div>
      ) : (
        /* ---- Active Player ---- */
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {/* Header row */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              {/* Play button */}
              <button
                onClick={handlePlayPause}
                style={{
                  width: 48, height: 48, borderRadius: '50%', border: 'none',
                  background: isPlaying
                    ? 'linear-gradient(135deg, rgba(228,187,124,0.25), rgba(228,187,124,0.15))'
                    : 'linear-gradient(135deg, rgba(228,187,124,0.2), rgba(228,187,124,0.1))',
                  color: 'var(--gold)', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'all 0.2s ease', flexShrink: 0,
                  boxShadow: isPlaying ? '0 0 20px rgba(228,187,124,0.15)' : 'none'
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'linear-gradient(135deg, rgba(228,187,124,0.35), rgba(228,187,124,0.2))'}
                onMouseLeave={e => e.currentTarget.style.background = isPlaying
                  ? 'linear-gradient(135deg, rgba(228,187,124,0.25), rgba(228,187,124,0.15))'
                  : 'linear-gradient(135deg, rgba(228,187,124,0.2), rgba(228,187,124,0.1))'}
              >
                {isPlaying ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                )}
              </button>
              <div>
                <h3 style={{
                  fontFamily: 'var(--font-serif)', fontSize: 16, fontWeight: 400,
                  color: 'var(--gold)', margin: 0, letterSpacing: '0.3px'
                }}>
                  {title}
                </h3>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '2px 0 0', opacity: 0.6 }}>
                  AI-generated audio discussion
                </p>
              </div>
            </div>
            <span style={{
              fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)',
              opacity: 0.5, letterSpacing: '0.5px'
            }}>
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          {/* Waveform visualization */}
          <div
            onClick={handleProgressClick}
            style={{
              height: 48, display: 'flex', alignItems: 'center', gap: 2,
              cursor: 'pointer', padding: '4px 0', position: 'relative'
            }}
          >
            {bars.map((h, i) => {
              const barProgress = (i / bars.length) * 100;
              const isActive = barProgress < progress;
              const isCurrent = Math.abs(barProgress - progress) < 2.5;
              const animHeight = isPlaying
                ? h * 40 + Math.sin(Date.now() / 200 + i * 0.5) * (isCurrent ? 6 : 2)
                : h * 36;
              return (
                <div key={i} style={{
                  flex: 1, height: `${animHeight}px`, borderRadius: 2,
                  background: isActive
                    ? isCurrent
                      ? 'var(--gold)'
                      : 'linear-gradient(180deg, rgba(228,187,124,0.8), rgba(228,187,124,0.4))'
                    : 'rgba(228, 187, 124, 0.12)',
                  transition: isPlaying ? 'none' : 'all 0.3s ease',
                  boxShadow: isCurrent ? '0 0 8px rgba(228,187,124,0.4)' : 'none'
                }} />
              );
            })}
          </div>

          {/* Progress bar (thin, below waveform) */}
          <div style={{
            height: 3, background: 'rgba(228,187,124,0.08)', borderRadius: 2,
            marginTop: 8, overflow: 'hidden', cursor: 'pointer'
          }} onClick={handleProgressClick}>
            <div style={{
              height: '100%', width: `${progress}%`,
              background: 'linear-gradient(90deg, var(--gold), rgba(228,187,124,0.6))',
              borderRadius: 2, transition: isPlaying ? 'none' : 'width 0.3s ease'
            }} />
          </div>
        </div>
      )}

      {/* Hidden audio element */}
      {activeSrc && (
        <audio
          ref={audioRef}
          key={activeSrc}
          src={activeSrc}
          preload="metadata"
          onLoadedMetadata={handleLoadedMetadata}
          onError={handleError}
          onEnded={handleEnded}
        />
      )}
    </div>
  );
}
