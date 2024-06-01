import React from 'react';

const ColorwheelIcon = (props: React.SVGAttributes<SVGElement>) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="1rem"
    height="1rem"
    viewBox="0 0 24 24"
    {...props}
  >
    <defs>
      <clipPath id="a" clipPathUnits="userSpaceOnUse">
        <ellipse
          cx={16.213}
          cy={5.334}
          rx={5.67}
          ry={9.368}
          style={{
            fill: 'none',
            stroke: '#FFF',
            strokeWidth: 0.285422,
          }}
          transform="rotate(27)"
        />
      </clipPath>
    </defs>
    <g clipPath="url(#a)">
      <path
        d="m12.073 12.098 4.228-8.305.02-.548-7.885.039L5.8 12.059Z"
        style={{
          fill: '#f55',
          fillOpacity: 1,
          stroke: 'none',
          strokeWidth: 0.199609,
          strokeOpacity: 1,
        }}
      />
      <path
        d="m12.046 12.157 4.323-8.48 2.429 2.304.197 6.176z"
        style={{
          fill: '#5555fd',
          fillOpacity: 1,
          stroke: 'none',
          strokeWidth: 0.201394,
          strokeOpacity: 1,
        }}
      />
      <path
        d="m12.073 12.093 6.745.02-3.657 7.728-7.394 1.612-.02-.924z"
        style={{
          fill: '#5a5',
          fillOpacity: 1,
          stroke: 'none',
          strokeWidth: 0.2,
          strokeOpacity: 1,
        }}
      />
      <path
        d="m12.069 12.104-6.263-.08-1.067 8.485h2.983Z"
        style={{
          fill: '#fa0',
          fillOpacity: 1,
          stroke: 'none',
          strokeWidth: 0.201053,
          strokeOpacity: 1,
        }}
      />
    </g>
    <ellipse
      cx={16.213}
      cy={5.334}
      rx={5.67}
      ry={9.368}
      style={{
        fill: 'none',
        stroke: '#FFF',
        strokeWidth: 0.285423,
      }}
      transform="rotate(27)"
    />
  </svg>
);

export default ColorwheelIcon;
