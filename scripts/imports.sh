python -X importtime -c "from speedy_utils import *" 2>&1 \
  | awk '
      /import time:/ {
        # second-last column looks like: 0.123>
        raw=$(NF-1)
        gsub(/[>]/,"",raw)
        if (raw > 900) print
      }
    '