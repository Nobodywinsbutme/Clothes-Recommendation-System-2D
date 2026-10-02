package com.group10.fashion.auth;

import com.group10.fashion.auth.AuthDtos.*;
import com.group10.fashion.security.JwtService;
import com.group10.fashion.user.User;
import com.group10.fashion.user.UserRepository;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final UserRepository users;
    private final PasswordEncoder encoder;
    private final JwtService jwt;

    public AuthController(UserRepository users, PasswordEncoder encoder, JwtService jwt) {
        this.users = users;
        this.encoder = encoder;
        this.jwt = jwt;
    }

    @PostMapping("/register")
    @ResponseStatus(HttpStatus.CREATED)
    public AuthResponse register(@Valid @RequestBody RegisterRequest req) {
        String email = req.email().trim().toLowerCase();
        if (users.existsByEmail(email)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Email đã được sử dụng");
        }
        User u = users.save(new User(email, encoder.encode(req.password()), req.displayName().trim()));
        return new AuthResponse(jwt.generateToken(u.getId(), u.getEmail()), u.getEmail(), u.getDisplayName());
    }

    @PostMapping("/login")
    public AuthResponse login(@Valid @RequestBody LoginRequest req) {
        User u = users.findByEmail(req.email().trim().toLowerCase())
            .filter(x -> encoder.matches(req.password(), x.getPasswordHash()))
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Email hoặc mật khẩu không đúng"));
        return new AuthResponse(jwt.generateToken(u.getId(), u.getEmail()), u.getEmail(), u.getDisplayName());
    }

    @GetMapping("/me")
    public MeResponse me(Authentication auth) {
        User u = users.findByEmail(auth.getName())
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED));
        return new MeResponse(u.getId(), u.getEmail(), u.getDisplayName());
    }
}
